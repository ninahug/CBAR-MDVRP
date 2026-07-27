#!/usr/bin/env python3
"""Stage 3 / E3: value of stochastic carbon-budget allocation (tex,
subsec:e3... and Table frozen_data_suites' E3 row).

For each of the 8 E3 instances and each nested training size (20/50/100,
prefixes of one 100-scenario bank -- see suite_config.build_e3_specs):
  1. Run PPBRC on the training bank to get an in-sample budget B* and
     in-sample objective.
  2. VSS = Z*_stochastic(B shared) - Z*_wait_and_see(B per scenario), both
     evaluated on the training scenarios' own emissions (budget_master.py,
     Layer 3, already exact).
  3. Out-of-sample: fix B* and re-solve routing (recourse-optimal, not
     re-adapting the budget) on held-out validation/test scenarios,
     reporting mean, standard error, and 95% CI.

Each held-out scenario requires its own routing re-solve at the fixed budget,
so out-of-sample evaluation dominates the cost of this experiment. The banks
were sized with that in mind (50 validation, 100 sealed test, 50 shift), and
by default the full bank is evaluated, as the paper states. --eval-subsample
caps the number of scenarios per bank for a quicker exploratory run; any value
other than 0 departs from the reported protocol and should not be used for
results that go into the paper.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import ppbrc
import budget_master
import recourse
from instance_io import load_instance


def out_of_sample_eval(inst, B, scenarios, alns_iterations, price_rounds, seed):
    D = [d["idx"] for d in inst.depots]
    values = []
    for w, sc in enumerate(scenarios):
        demand = inst.scenario_demand(sc)
        result = ppbrc.solve_scenario(inst, demand, B, price_rounds=price_rounds,
                                       alns_iterations=alns_iterations, seed=seed + w,
                                       price_guided=True, diversify=False)
        g = {d: result["emissions"][d] - B[d] for d in D}
        rec = recourse.solve_recourse_primal(g, D, inst.transfer_arcs, inst.prices["buy"],
                                              inst.prices["sell"])["value"]
        values.append(result["distance"] + rec)
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    se = math.sqrt(var / n)
    return {"mean": mean, "se": se, "ci95": [mean - 1.96 * se, mean + 1.96 * se], "n_evaluated": n}


def run_one(path: str, eval_subsample: int, alns_iterations: int, price_rounds: int,
            outer_rounds: int, seed: int) -> dict:
    inst = load_instance(path)
    D = [d["idx"] for d in inst.depots]
    pi_train = {}
    row = {"name": inst.name, "n_customers": inst.n_customers, "n_depots": inst.n_depots}

    for train_size in (20, 50, 100):
        train_scenarios = inst.scenarios("train", prefix=train_size)
        pi = {w: train_scenarios[w]["probability"] for w in range(len(train_scenarios))}

        t0 = time.time()
        ppbrc_result = ppbrc.run_ppbrc(inst, train_scenarios, outer_rounds=outer_rounds,
                                        price_rounds=price_rounds, alns_iterations=alns_iterations, seed=seed)
        B_star = ppbrc_result["B"]
        in_sample_obj = ppbrc_result["objective"]

        vss_result = budget_master.value_of_stochastic_solution(
            ppbrc_result["emissions"], pi, D, inst.transfer_arcs, inst.prices,
            inst.depot_budget_bounds, inst.corporate_budget)

        limit = eval_subsample or None   # 0 means evaluate the whole bank
        val_scenarios = inst.scenarios("validation")[:limit]
        test_scenarios = inst.scenarios("test")[:limit]
        eval_iterations = max(15, alns_iterations // 4)
        val_eval = out_of_sample_eval(inst, B_star, val_scenarios, eval_iterations, 1, seed + 1)
        test_eval = out_of_sample_eval(inst, B_star, test_scenarios, eval_iterations, 1, seed + 2)
        runtime = time.time() - t0

        row[f"train{train_size}"] = {
            "in_sample_objective": in_sample_obj,
            "vss": vss_result["vss"],
            "stochastic_value": vss_result["stochastic"],
            "wait_and_see_value": vss_result["wait_and_see"],
            "validation": val_eval,
            "test": test_eval,
            "runtime": runtime,
        }
        print(f"  train={train_size}: in_sample={in_sample_obj:.4f} VSS={vss_result['vss']:.4f} "
              f"val_mean={val_eval['mean']:.4f}(+-{val_eval['se']:.4f}) "
              f"test_mean={test_eval['mean']:.4f}(+-{test_eval['se']:.4f}) [{runtime:.0f}s]", flush=True)

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\CBAR_MDVRP\data\generated\synthetic_v1\E3")
    ap.add_argument("--out", default=r"D:\CBAR_MDVRP\data\results\stage3_e3\stage3_e3_results.jsonl")
    ap.add_argument("--eval-subsample", type=int, default=0,
                     help="0 evaluates the full validation and test banks, as reported; "
                          "a positive value truncates them for exploratory runs")
    ap.add_argument("--alns-iterations", type=int, default=60)
    ap.add_argument("--price-rounds", type=int, default=2)
    ap.add_argument("--outer-rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=91001)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.data) / "*.json")))
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])

    with out_path.open("a", encoding="utf-8") as f:
        for path in files:
            name = Path(path).stem
            if name in done:
                print(f"skip {name}")
                continue
            print(f"running {name} ...", flush=True)
            row = run_one(path, args.eval_subsample, args.alns_iterations, args.price_rounds,
                           args.outer_rounds, args.seed)
            f.write(json.dumps(row) + "\n")
            f.flush()

    print("done.")


if __name__ == "__main__":
    main()

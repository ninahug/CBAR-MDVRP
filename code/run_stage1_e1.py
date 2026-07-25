#!/usr/bin/env python3
"""Stage 1 / E1: exact MILP vs. PPBRC on all 36 E1 instances.

Writes results incrementally to --out (JSON lines) so partial progress
survives an interruption. Each line: instance name, sizes, exact status/
objective/bound/integer-feasibility/proven flag/runtime, PPBRC objective/
runtime, and gaps to both the best incumbent and the best proven bound (the
tex explicitly allows "a proven optimum or the best valid bound within the
stated time limit" -- E1's arc-flow MILP is not expected to prove
optimality, or even find an integer-feasible incumbent, on every cell; all
of this is reported honestly rather than only the flattering number).

`integer_feasible=False` (CBC never found an integer solution in time, so
exact_model.py reports objective=None) is common at C=15/20, W=10 -- in that
case only the bound-validity sanity check applies: PPBRC's feasible
objective must be >= the proven bound (minus tolerance). A violation is a
real bug flag, not a heuristic quality issue, since the bound is valid
regardless of whether an incumbent was ever found.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from instance_io import load_instance
import exact_model
import ppbrc

BOUND_VIOLATION_TOL = 1e-3


def run_one(path: str, time_limit: float, alns_iterations: int) -> dict:
    inst = load_instance(path)
    scenarios = inst.scenarios("train")
    t0 = time.time()
    exact = exact_model.build_and_solve(inst, scenarios, time_limit=time_limit, msg=False)
    exact_runtime = time.time() - t0

    t0 = time.time()
    result = ppbrc.run_ppbrc(inst, scenarios, outer_rounds=3, price_rounds=2,
                              alns_iterations=alns_iterations, seed=91001)
    ppbrc_runtime = time.time() - t0
    ppbrc_obj = result["objective"]

    exact_obj = exact["objective"]
    exact_bound = exact["bound"]
    gap_to_incumbent = ((ppbrc_obj - exact_obj) / max(1.0, abs(exact_obj))
                         if exact_obj is not None else None)
    bound_violated = exact_bound is not None and ppbrc_obj < exact_bound - BOUND_VIOLATION_TOL

    return {
        "name": inst.name,
        "n_customers": inst.n_customers,
        "n_depots": inst.n_depots,
        "n_scenarios": len(scenarios),
        "exact_status": exact["status"],
        "exact_integer_feasible": exact["integer_feasible"],
        "exact_objective": exact_obj,
        "exact_bound": exact_bound,
        "exact_proven_optimal": exact["proven_optimal"],
        "exact_runtime": exact_runtime,
        "ppbrc_objective": ppbrc_obj,
        "ppbrc_runtime": ppbrc_runtime,
        "ppbrc_outer_rounds_used": result["outer_rounds_used"],
        "gap_to_incumbent_pct": gap_to_incumbent * 100 if gap_to_incumbent is not None else None,
        "bound_violated": bound_violated,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\CBAR_MDVRP\data\generated\synthetic_v1\E1")
    ap.add_argument("--out", default=r"D:\CBAR_MDVRP\data\results\stage1_e1\stage1_e1_results.jsonl")
    ap.add_argument("--time-limit", type=float, default=90.0)
    ap.add_argument("--alns-iterations", type=int, default=200)
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
                print(f"skip {name} (already done)")
                continue
            print(f"running {name} ...", flush=True)
            row = run_one(path, args.time_limit, args.alns_iterations)
            f.write(json.dumps(row) + "\n")
            f.flush()
            exact_str = (f"{row['exact_objective']:.4f}" if row["exact_objective"] is not None
                         else f"NO-INTEGER-SOLUTION(max_frac)")
            gap_str = (f"{row['gap_to_incumbent_pct']:.2f}%" if row["gap_to_incumbent_pct"] is not None
                       else "n/a")
            print(f"  exact={exact_str} (bound={row['exact_bound']:.4f}, "
                  f"int_feasible={row['exact_integer_feasible']}, proven={row['exact_proven_optimal']})  "
                  f"ppbrc={row['ppbrc_objective']:.4f}  gap={gap_str}  "
                  f"bound_violated={row['bound_violated']}  "
                  f"[{row['exact_runtime']:.0f}s / {row['ppbrc_runtime']:.0f}s]", flush=True)

    print("done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sensitivity analysis of recourse interaction (paper Section 6.5) over the
32 instances of the sensitivity suite.

Each instance is solved under the four recourse regimes the paper defines,
obtained by enabling or disabling the two channels:

  operational recourse  -- disabled by restricting every customer to its home
                           depot, so service cannot be moved between depots;
  accounting recourse   -- disabled by removing every internal transfer arc,
                           so carbon responsibility cannot be moved.

External trading remains available in all four regimes, so complete recourse
is preserved and the regimes stay comparable. Writing Z_ab for the expected
cost, with a and b indicating whether operational and accounting recourse are
available, the script reports

  V_op    = Z_00 - Z_10        value of operational recourse alone
  V_acc   = Z_00 - Z_01        value of accounting recourse alone
  V_joint = Z_00 - Z_11        value of both together
  I       = V_joint - V_op - V_acc      interaction

A positive I means the channels are complementary relative to additivity, a
negative I that they substitute. The factor levels of each instance are
carried through to the output so that effects on I can be estimated over the
replicated 2^4 design without re-deriving the design matrix.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import time
from pathlib import Path

from instance_io import Instance, load_instance
import methods


def apply_regime(inst: Instance, operational: bool, accounting: bool) -> Instance:
    """A copy of `inst` restricted to one of the four recourse regimes.

    Disabling operational recourse restricts each customer's eligible depot
    set to its home depot; disabling accounting recourse empties the transfer
    network. Neither touches external trading, which remains available so that
    every regime admits a feasible second stage.
    """
    m = copy.copy(inst)
    if not operational:
        m.eligible = {c["idx"]: [c["home_depot"]] for c in inst.customers}
        m.customers_of_vehicle = {
            v.idx: [c["idx"] for c in inst.customers if c["home_depot"] == v.depot]
            for v in inst.vehicles
        }
    if not accounting:
        m.transfer_arcs = []
    return m


REGIMES = [
    ("Z00", False, False),
    ("Z10", True, False),
    ("Z01", False, True),
    ("Z11", True, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\CBAR_MDVRP\data\generated\synthetic_v1\E4")
    ap.add_argument("--out", default=r"D:\CBAR_MDVRP\data\results\stage4_e4\stage4_e4_results.jsonl")
    ap.add_argument("--method", default="PPBRC")
    ap.add_argument("--seed", type=int, default=91001)
    ap.add_argument("--outer-rounds", type=int, default=3)
    ap.add_argument("--price-rounds", type=int, default=2)
    ap.add_argument("--alns-iterations", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.data) / "*.json")))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["instance"])

    with out_path.open("a", encoding="utf-8") as f:
        for idx, path in enumerate(files):
            if idx % args.num_shards != args.shard:
                continue
            inst = load_instance(path)
            if inst.name in done:
                continue

            print(f"[shard {args.shard}] {inst.name} ...", flush=True)
            scenarios = inst.scenarios("train")
            Z, per_regime = {}, {}
            t0 = time.time()
            for tag, operational, accounting in REGIMES:
                restricted = apply_regime(inst, operational, accounting)
                result = methods.run_method(
                    args.method, restricted, scenarios, seed=args.seed,
                    outer_rounds=args.outer_rounds, price_rounds=args.price_rounds,
                    alns_iterations=args.alns_iterations)
                Z[tag] = result["objective"]
                diag = methods.mechanism_diagnostics(restricted, result)
                per_regime[tag] = {
                    "objective": result["objective"],
                    "distance": result["distance"],
                    "recourse_cost": result["recourse_cost"],
                    "cross_depot_share_pct": cross_depot_share(restricted, result),
                    **diag,
                }
                print(f"    {tag}: {result['objective']:.4f}", flush=True)
            runtime = time.time() - t0

            V_op = Z["Z00"] - Z["Z10"]
            V_acc = Z["Z00"] - Z["Z01"]
            V_joint = Z["Z00"] - Z["Z11"]
            row = {
                "instance": inst.name,
                "n_customers": inst.n_customers,
                "n_depots": inst.n_depots,
                "factors": inst.raw["factors"],
                "method": args.method,
                "seed": args.seed,
                "Z": Z,
                "V_op": V_op,
                "V_acc": V_acc,
                "V_joint": V_joint,
                "interaction_I": V_joint - V_op - V_acc,
                "regimes": per_regime,
                "runtime": runtime,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"  V_op={V_op:.4f} V_acc={V_acc:.4f} V_joint={V_joint:.4f} "
                  f"I={row['interaction_I']:.4f}  [{runtime:.0f}s]", flush=True)

    print("done.")


def cross_depot_share(inst: Instance, result: dict) -> float:
    """Percentage of served customers assigned to a depot other than their
    home depot, averaged over scenarios. Zero by construction when
    operational recourse is disabled."""
    total = served_away = 0
    home = {c["idx"]: c["home_depot"] for c in inst.customers}
    for sol in result["solutions"].values():
        for depot, routes in sol.routes.items():
            for route in routes:
                for i in route:
                    total += 1
                    if home[i] != depot:
                        served_away += 1
    return 100.0 * served_away / total if total else 0.0


if __name__ == "__main__":
    main()

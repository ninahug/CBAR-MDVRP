#!/usr/bin/env python3
"""At what carbon price does operational recourse switch on?

The model offers two ways to absorb a depot's carbon imbalance: move service,
by reassigning customers and rerouting, or move responsibility, by
transferring credit. Moving service changes physical operations and is
therefore only worth doing when the carbon term is large enough to outweigh
the routing cost it incurs. That threshold is a property of the instance and
the prices, not of the algorithm, and it is what a firm needs to know before
delegating carbon budgets at all.

This experiment scans the external price level, holding the purchase-to-sale
ratio fixed so that the spread structure is unchanged, and records how far the
solution departs from the pure-routing solution: the share of customers served
away from their home depot, the use of the transfer network, and the share of
total cost attributable to carbon. Reported against the price level, these
identify the range over which the mechanism is inactive, the range where it
switches on, and the range where it dominates routing.

The price level is reported both in the model's own units and as the ratio of
carbon cost to travel cost, which is the quantity that transfers across
applications: a fleet whose carbon cost is a few per cent of operating cost
sits in a different regime from one where it is comparable.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import time
from pathlib import Path

import carbon
import methods
from instance_io import load_instance


def scaled_instance(inst, scale: float):
    """The same instance with both external prices scaled. The ratio
    p_buy/p_sell is preserved, so the spread that rules out simultaneous
    purchase and sale is unchanged and only the level moves."""
    m = copy.copy(inst)
    m.prices = {"buy": inst.prices["buy"] * scale,
                "sell": inst.prices["sell"] * scale}
    return m


def carbon_cost_ratio(inst, scale: float) -> float:
    """Carbon cost per unit distance at the mean price, divided by travel cost
    per unit distance. Travel cost is one per unit distance in these
    instances, and emission per unit distance lies between the empty and full
    load rates, so the midpoint is used."""
    mean_price = 0.5 * (inst.prices["buy"] + inst.prices["sell"]) * scale
    mean_rate = carbon.THETA * 0.5 * (carbon.RHO_EMPTY + carbon.RHO_FULL)
    return mean_price * mean_rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\CBAR_MDVRP\data\generated\synthetic_v1\E4")
    ap.add_argument("--out", default=r"D:\CBAR_MDVRP\data\results\price_scan\price_scan_results.jsonl")
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--scales", type=float, nargs="*",
                     default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--method", default="PPBRC")
    ap.add_argument("--seed", type=int, default=91001)
    ap.add_argument("--scenarios", type=int, default=10)
    ap.add_argument("--alns-iterations", type=int, default=80)
    ap.add_argument("--outer-rounds", type=int, default=2)
    ap.add_argument("--price-rounds", type=int, default=2)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["instance"], r["price_scale"]))

    files = sorted(glob.glob(str(Path(args.data) / "*.json")))[:args.instances]
    with out_path.open("a", encoding="utf-8") as f:
        for path in files:
            base = load_instance(path)
            scenarios = base.scenarios("train")[:args.scenarios]
            home = {c["idx"]: c["home_depot"] for c in base.customers}
            for scale in args.scales:
                if (base.name, scale) in done:
                    continue
                inst = scaled_instance(base, scale)
                t0 = time.time()
                res = methods.run_method(args.method, inst, scenarios, seed=args.seed,
                                          outer_rounds=args.outer_rounds,
                                          price_rounds=args.price_rounds,
                                          alns_iterations=args.alns_iterations)
                runtime = time.time() - t0

                total = away = 0
                for sol in res["solutions"].values():
                    for depot, routes in sol.routes.items():
                        for route in routes:
                            for i in route:
                                total += 1
                                if home[i] != depot:
                                    away += 1
                diag = methods.mechanism_diagnostics(inst, res)
                obj = res["objective"]
                row = {
                    "instance": base.name,
                    "price_scale": scale,
                    "p_buy": inst.prices["buy"],
                    "p_sell": inst.prices["sell"],
                    "carbon_to_travel_ratio": carbon_cost_ratio(base, scale),
                    "objective": obj,
                    "distance": res["distance"],
                    "recourse_cost": res["recourse_cost"],
                    "carbon_cost_share": (res["recourse_cost"] / obj if abs(obj) > 1e-9 else None),
                    "cross_depot_share_pct": 100.0 * away / total if total else 0.0,
                    "runtime": runtime,
                    **diag,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{base.name:20s} scale={scale:<6g} ratio={row['carbon_to_travel_ratio']:8.3f} "
                      f"cross-depot={row['cross_depot_share_pct']:5.1f}%  "
                      f"transfer-use={diag['transfer_use_pct']:5.1f}%  [{runtime:.0f}s]",
                      flush=True)

    print("done.")


if __name__ == "__main__":
    main()

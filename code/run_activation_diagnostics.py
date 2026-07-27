#!/usr/bin/env python3
"""Mechanism-activation diagnostics at the reference allocation.

Fills Table "mechanism_activation" of the tex: an instance can only
discriminate between methods if the rebalancing mechanism is actually live on
it -- if no depot ever falls short while another is long, or if no transfer arc
ever carries flow, the instance says nothing about the coordination scheme
however carefully it is solved.

Under the inventory model this is cheap to compute and needs no routing solve
at all. The stock a depot draws is the demand assigned to it (eq:c20), so at
the reference assignment -- every customer served by its home depot -- the
scenario deficit vector follows directly from the scenario demands, and only
the recourse LP has to be solved.

Usage:
    python run_activation_diagnostics.py                  # every suite
    python run_activation_diagnostics.py --suites E2 E4
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import instance_io
import recourse

TOL = 1e-7


def classify(inst, scenario) -> dict:
    """Diagnostics for one scenario at the reference allocation."""
    depots = [d["idx"] for d in inst.depots]
    alloc = inst.depot_budget_allocation
    demand = inst.scenario_demand(scenario)

    drawn = {d: 0.0 for d in depots}
    for c in inst.customers:
        drawn[c["home_depot"]] += demand[c["idx"]]

    g = {d: drawn[d] - alloc.get(d, 0.0) for d in depots}
    sol = recourse.solve_recourse(g, depots, inst.transfer_arcs, inst.prices)
    primal, dual = sol["primal"], sol["dual"]

    lam = dual["lambda"]
    transferred = sum(primal["t"].values())
    saturated = sum(
        1 for a in inst.transfer_arcs
        if primal["t"].get((a["from"], a["to"]), 0.0) >= a["capacity"] - TOL
        and a["capacity"] > TOL
    )
    return {
        "coexist": any(v > TOL for v in g.values()) and any(v < -TOL for v in g.values()),
        "dispersion": max(lam.values()) - min(lam.values()) if lam else 0.0,
        "transfer_used": transferred > TOL,
        "saturated_frac": saturated / len(inst.transfer_arcs) if inst.transfer_arcs else 0.0,
        "purchased": sum(primal["b"].values()) > TOL,
        "sold": sum(primal["s"].values()) > TOL,
        "duality_gap": sol["duality_gap"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\CBAR_MDVRP\data\generated\synthetic_v1")
    ap.add_argument("--suites", nargs="*", default=None)
    ap.add_argument("--out", default=r"D:\CBAR_MDVRP\data\results\activation_diagnostics.json")
    args = ap.parse_args()

    root = Path(args.data)
    suites = args.suites or sorted(p.name for p in root.iterdir() if p.is_dir())

    rows = {}
    for suite in suites:
        by_class: dict[str, list[dict]] = defaultdict(list)
        for path in sorted((root / suite).glob("*.json")):
            inst = instance_io.load_instance(path)
            # E2 and E4 are stratified; report per stratum, others in one block.
            factors = inst.raw.get("factors", {})
            if suite == "E2":
                key = "%s  (gamma_B=%.2f, tau=%.2f)" % (
                    suite, factors.get("budget_factor", 0), factors.get("capacity_scale", 0))
            else:
                key = suite
            for scenario in inst.scenarios("train"):
                by_class[key].append(classify(inst, scenario))

        for key, recs in sorted(by_class.items()):
            n = len(recs)
            rows[key] = {
                "scenarios": n,
                "coexistence_pct": 100.0 * sum(r["coexist"] for r in recs) / n,
                "mean_price_dispersion": statistics.fmean(r["dispersion"] for r in recs),
                "transfer_use_pct": 100.0 * sum(r["transfer_used"] for r in recs) / n,
                "saturated_arcs_pct": 100.0 * statistics.fmean(r["saturated_frac"] for r in recs),
                "purchase_pct": 100.0 * sum(r["purchased"] for r in recs) / n,
                "sale_pct": 100.0 * sum(r["sold"] for r in recs) / n,
                "max_duality_gap": max(r["duality_gap"] for r in recs),
            }

    hdr = ("class", "scen", "coexist%", "disp", "transfer%", "sat%", "buy%", "sell%")
    print("%-34s %5s %9s %7s %10s %6s %6s %6s" % hdr)
    for key, r in rows.items():
        print("%-34s %5d %9.1f %7.3f %10.1f %6.1f %6.1f %6.1f" % (
            key, r["scenarios"], r["coexistence_pct"], r["mean_price_dispersion"],
            r["transfer_use_pct"], r["saturated_arcs_pct"],
            r["purchase_pct"], r["sale_pct"]))

    worst_gap = max((r["max_duality_gap"] for r in rows.values()), default=0.0)
    print("\nmax duality gap across all recourse solves: %.2e" % worst_gap)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

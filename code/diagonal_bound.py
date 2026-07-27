#!/usr/bin/env python3
"""Feasibility probe for a diagonal Lagrangian lower bound.

The extended price polyhedron contains every constant vector, since a
constant price makes each arbitrage constraint read 0 <= kappa. Fixing
lambda = lambda_bar * 1 and mu = 0 in the dual representation therefore gives,
for any lambda_bar in [p_sell, p_buy],

    Psi_Tbar(E - B) >= lambda_bar * ( sum_d E_d - B_corp ),

and hence

    Z >= max_lambda_bar { sum_w pi_w * min_S [ C_op(S) + lambda_bar * sum_d E_d(S) ]
                          - lambda_bar * B_corp }.

The inner problem is a multi-depot routing problem with a uniform emission
surcharge: no per-depot price, no budget, no transfer network. Any lower bound
on it yields a valid lower bound on the two-stage problem, and the outer
maximisation is one-dimensional.

This script does not implement a strong bound for the inner problem. It
computes a deliberately simple one -- a two-cheapest-incident-edges bound on
distance, plus the unavoidable payload-distance of delivering each demand from
its nearest depot -- in order to answer a single question before any effort is
invested: is a bound of this shape anywhere near tight enough to be worth
strengthening?
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import carbon
from instance_io import load_instance


def distance_lower_bound(inst, demand) -> float:
    """Every served customer is entered once and left once, so the total
    distance is at least half the sum, over customers, of the two cheapest
    incident edges available to it."""
    nodes = [(c["idx"], c["x"], c["y"]) for c in inst.customers]
    depots = [(d["idx"], d["x"], d["y"]) for d in inst.depots]
    total = 0.0
    for i, xi, yi in nodes:
        ds = [math.hypot(xi - x, yi - y) for _, x, y in depots]
        ds += [math.hypot(xi - x, yi - y) for j, x, y in nodes if j != i]
        ds.sort()
        total += 0.5 * (ds[0] + ds[1])
    return total


def payload_distance_lower_bound(inst, demand) -> float:
    """The demand of customer i is carried on every arc between its depot and
    i, so its contribution to sum(d_ij f_ij) is at least q_i times the
    distance from the nearest depot."""
    total = 0.0
    for c in inst.customers:
        nearest = min(math.hypot(c["x"] - d["x"], c["y"] - d["y"]) for d in inst.depots)
        total += demand[c["idx"]] * nearest
    return total


def inner_lower_bound(inst, demand, lam_bar: float) -> float:
    """Lower bound on min_S [ C_op(S) + lam_bar * sum_d E_d(S) ] for one
    scenario, with the emission model of carbon.py."""
    theta, rho0, rhoQ = carbon.THETA, carbon.RHO_EMPTY, carbon.RHO_FULL
    cap = min(d["capacity"] for d in inst.depots)
    arc_coeff = 1.0 + lam_bar * theta * rho0
    load_coeff = lam_bar * theta * (rhoQ - rho0) / cap
    return (arc_coeff * distance_lower_bound(inst, demand)
            + load_coeff * payload_distance_lower_bound(inst, demand))


def diagonal_bound(inst, scenarios, grid: int = 41) -> tuple[float, float]:
    """Maximise the diagonal bound over lambda_bar on a grid. Returns the best
    bound and the price attaining it."""
    p_sell, p_buy = inst.prices["sell"], inst.prices["buy"]
    best, best_lam = -float("inf"), p_sell
    for k in range(grid):
        lam_bar = p_sell + (p_buy - p_sell) * k / (grid - 1)
        value = 0.0
        for sc in scenarios:
            demand = inst.scenario_demand(sc)
            value += sc["probability"] * inner_lower_bound(inst, demand, lam_bar)
        value -= lam_bar * inst.corporate_budget
        if value > best:
            best, best_lam = value, lam_bar
    return best, best_lam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\CBAR_MDVRP\data\generated\synthetic_v1\E1")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=120.0)
    args = ap.parse_args()

    import exact_model

    print(f"{'instance':22s} {'exact':>10s} {'bound':>10s} {'gap %':>8s} {'lam*':>6s}")
    print("-" * 60)
    gaps = []
    for path in sorted(glob.glob(str(Path(args.data) / "*.json")))[:args.limit]:
        inst = load_instance(path)
        scenarios = inst.scenarios("train")
        res = exact_model.build_and_solve(inst, scenarios, time_limit=args.time_limit)
        if not res["integer_feasible"]:
            print(f"{inst.name:22s} {'no incumbent':>10s}")
            continue
        exact = res["objective"]
        lb, lam_star = diagonal_bound(inst, scenarios)
        gap = (exact - lb) / max(abs(exact), 1e-9) * 100.0
        gaps.append(gap)
        print(f"{inst.name:22s} {exact:10.4f} {lb:10.4f} {gap:8.1f} {lam_star:6.2f}")

    if gaps:
        print("-" * 60)
        print(f"mean gap {sum(gaps)/len(gaps):.1f}%   best {min(gaps):.1f}%   worst {max(gaps):.1f}%")


if __name__ == "__main__":
    main()

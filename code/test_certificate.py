#!/usr/bin/env python3
"""Randomised check of the critical-region certificate in recourse.py.

The certificate claims that when it fires,

    Psi(g + delta) - Psi(g) = lambda^T delta

holds exactly. That is the claim the search relies on, so it is tested against
a full re-solve of the recourse problem rather than assumed. The test also
records how often the certificate fires, since a certificate that were never
satisfied would be sound but useless, and reports the worst violation of the
subgradient bound, which must hold whether or not the certificate fires.
"""
from __future__ import annotations

import random
import sys

import recourse


def random_network(rng, n_depots):
    """A bidirected ring. For two depots the ring is the single edge {0,1};
    iterating d -> (d+1) mod n would emit that edge twice."""
    depots = list(range(n_depots))
    edges = [(0, 1)] if n_depots == 2 else [(d, (d + 1) % n_depots) for d in range(n_depots)]
    arcs = []
    for d, e in edges:
        kappa = rng.uniform(0.05, 1.2)
        cap = rng.uniform(2.0, 40.0)
        arcs.append({"from": d, "to": e, "friction": kappa, "capacity": cap})
        arcs.append({"from": e, "to": d, "friction": kappa * rng.uniform(0.8, 1.2),
                     "capacity": cap})
    return depots, arcs


def main(trials: int = 400, seed: int = 20260726) -> int:
    rng = random.Random(seed)
    p_buy, p_sell = 9.0, 2.0
    certified = 0
    worst_certified_error = 0.0
    worst_bound_violation = 0.0
    worst_magnitude = 0.0
    checked = 0

    for _ in range(trials):
        n_depots = rng.choice([2, 3, 4, 5])
        depots, arcs = random_network(rng, n_depots)
        g = {d: rng.uniform(-30, 30) for d in depots}

        base = recourse.solve_recourse(g, depots, arcs, {"buy": p_buy, "sell": p_sell})
        lam = base["dual"]["lambda"]
        r = recourse.net_external_position(base["primal"], depots)
        psi_g = base["primal"]["value"]

        for _ in range(3):
            scale = rng.choice([0.05, 0.5, 5.0])
            delta = {d: rng.uniform(-scale, scale) for d in depots}
            g2 = {d: g[d] + delta[d] for d in depots}
            psi_g2 = recourse.solve_recourse_primal(
                g2, depots, arcs, p_buy, p_sell)["value"]

            worst_magnitude = max(worst_magnitude, abs(psi_g), abs(psi_g2))
            actual = psi_g2 - psi_g
            predicted = sum(lam[d] * delta[d] for d in depots)
            checked += 1

            # The subgradient bound must hold unconditionally.
            worst_bound_violation = max(worst_bound_violation, predicted - actual)

            if recourse.certify_same_region(lam, r, delta, p_buy, p_sell):
                certified += 1
                worst_certified_error = max(worst_certified_error,
                                            abs(actual - predicted))

    rate = 100.0 * certified / checked
    print(f"perturbations checked      : {checked}")
    print(f"certified                  : {certified} ({rate:.1f}%)")
    print(f"max |exact - predicted| on certified moves : {worst_certified_error:.3e}")
    print(f"max violation of the subgradient bound     : {worst_bound_violation:.3e}")
    print(f"largest |Psi| encountered                  : {worst_magnitude:.3e}")

    # Both quantities are compared against the solver's own accuracy rather
    # than an absolute constant. The subgradient bound is exact in theory, so
    # any violation of it measures the LP solver's error; the certificate is
    # sound only if its error is of the same order and not larger.
    solver_scale = max(worst_magnitude, 1.0)
    rel_certified = worst_certified_error / solver_scale
    rel_bound = worst_bound_violation / solver_scale
    print(f"relative to that magnitude: certified {rel_certified:.2e}, bound {rel_bound:.2e}")

    ok = (rel_certified < 1e-6 and rel_bound < 1e-6 and certified > 0
          and worst_certified_error <= worst_bound_violation * 1.001 + 1e-12)
    print("PASS" if ok else "FAIL")
    if not ok:
        print("  (a certified error materially exceeding the bound violation would"
              " indicate a logic error rather than solver noise)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

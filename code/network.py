"""Inter-depot stock transfer network.

A bidirectional ring (plus two chords when |D| = 5), directionally asymmetric
handling costs, and a capacity proportional to the reference demand of the
arc's two incident depots.

Two constants here are calibration choices rather than values the tex fixes
independently, and both are documented as ASSUMPTION.

ASYMMETRY is a fixed +/-15% split around the distance-based base cost, chosen
only to make the two directions of every arc numerically distinct and
reproducible from the depot indices alone.

CAPACITY_FRAC exists because capacity has to be scaled against the quantity
that actually crosses the network, and that is the *imbalance*, not the level.
The reference demand R^0_d is a depot's whole expected requirement, whereas the
volume needing transfer is the deviation of realised demand from it, netted
across depots. Measured over the E2 and E4 suites, the transferable imbalance
averages about 12% of the mean reference demand of an arc's two depots and
never exceeds 62%. Scaling capacity to R^0 directly therefore left every arc
slack in every scenario of every suite, which silently made the transfer-
capacity factor tau inert: the 2^4 sensitivity design would have reported a
null effect for tau that was an artefact of the calibration rather than an
economic finding, and the capacity multipliers of
Proposition~\\ref{prop:complementarity} would never have activated. Folding in
0.12 puts tau = 1.00 at roughly the median imbalance, so tau = 0.60 is
genuinely tight and tau = 1.40 genuinely slack.
"""
from __future__ import annotations

import math

FRICTION_BASE = 0.15
FRICTION_SLOPE = 0.35
ASYMMETRY = 0.15        # ASSUMPTION: see module docstring
CAPACITY_FRAC = 0.12    # ASSUMPTION: see module docstring


def build_transfer_network(depots, adjacency: list[tuple[int, int]],
                            capacity_scale: float, ref_by_depot: dict) -> list[dict]:
    if not adjacency:
        return []
    dist = {}
    for (a, b) in adjacency:
        da, db = depots[a], depots[b]
        d_ab = math.hypot(da.x - db.x, da.y - db.y)
        dist[(a, b)] = d_ab
        dist[(b, a)] = d_ab
    scale = sum(dist.values()) / len(dist)

    arcs = []
    for (a, b) in adjacency:
        d_ab = dist[(a, b)]
        base_kappa = FRICTION_BASE + FRICTION_SLOPE * d_ab / scale
        mean_ref = 0.5 * (ref_by_depot[a] + ref_by_depot[b])
        cap = capacity_scale * CAPACITY_FRAC * mean_ref
        arcs.append({"from": a, "to": b, "distance": d_ab,
                      "friction": base_kappa * (1 + ASYMMETRY), "capacity": cap})
        arcs.append({"from": b, "to": a, "distance": d_ab,
                      "friction": base_kappa * (1 - ASYMMETRY), "capacity": cap})
    return arcs

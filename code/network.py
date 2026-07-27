"""Internal carbon-credit transfer network.

Implements the "Carbon parameters and transfer network" paragraph: a
bidirectional ring (plus two chords when |D| = 5), directionally asymmetric
friction, and capacity "proportional to the mean reference emission of its
two incident depots and the pre-specified capacity scale."

The tex names directional asymmetry as a design requirement but does not
give a formula for it; ASYMMETRY below is an ASSUMPTION (a fixed +/-15%
split around the distance-based base friction) chosen only to make the two
directions of every arc numerically distinct and reproducible from the
depot indices alone.
"""
from __future__ import annotations

import math

FRICTION_BASE = 0.15
FRICTION_SLOPE = 0.35
ASYMMETRY = 0.15  # ASSUMPTION: see module docstring


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
        cap = capacity_scale * mean_ref
        arcs.append({"from": a, "to": b, "distance": d_ab,
                      "friction": base_kappa * (1 + ASYMMETRY), "capacity": cap})
        arcs.append({"from": b, "to": a, "distance": d_ab,
                      "friction": base_kappa * (1 - ASYMMETRY), "capacity": cap})
    return arcs

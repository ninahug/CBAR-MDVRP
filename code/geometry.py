"""Depot/customer geometry and fleet sizing.

Implements the "Geometry and eligibility" paragraph of Appendix
"Instance and scenario generation" in
t-mdvrp_CBAR_PPBRC_public_benchmark_protocol.tex: depots on a perturbed
regular polygon in a 10x10 region; core customers from depot-centred
Gaussian clusters; boundary customers around the midpoint of an adjacent
depot pair, eligible for exactly that pair.

The tex prose does not pin down several numeric constants (polygon jitter
magnitude, cluster spread, target fleet utilisation, vehicle capacity). Those
are marked ASSUMPTION below: documented modelling choices, not values quoted
from the paper. They are recorded in every instance's companion JSON so they
are auditable and can be revised without touching the generation logic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

REGION = 10.0

# ASSUMPTION: not specified numerically in the tex.
POLYGON_JITTER_FRAC = 0.05      # depot radial jitter, fraction of region side
POLYGON_ANGLE_JITTER = 0.30     # depot angular jitter, fraction of half the
                                 # inter-depot angle
CLUSTER_STD_FRAC = 0.12         # customer cluster std, fraction of mean
                                 # inter-depot spacing
TARGET_UTILIZATION = 0.62       # mean fleet-load target used to size vehicle
                                 # counts; kept well under 1 so that the
                                 # per-scenario individual-vehicle-feasibility
                                 # truncation rule (the only one the tex
                                 # specifies) essentially never needs an
                                 # aggregate-fleet fallback
VEHICLE_CAPACITY = 100.0        # homogeneous per the tex's remark that fleet
                                 # heterogeneity is "a modelling capability
                                 # rather than a separate contribution"
BASE_MEAN_DEMAND = 12.0         # target mean of q_bar_i before depot scaling
BASE_DEMAND_CV = 0.30           # heterogeneity of q_bar_i within a depot
DEPOT_SCALE_CV = 0.25           # heterogeneity of the depot-level demand
                                 # scale itself (tex: "Depot demand scales
                                 # are heterogeneous and are generated
                                 # independently of the scenario streams")


@dataclass
class Depot:
    idx: int
    x: float
    y: float
    capacity: float
    n_vehicles: int
    max_duration: float


@dataclass
class Customer:
    idx: int
    x: float
    y: float
    service: float
    base_demand: float
    home_depot: int
    eligible_depots: tuple


def euclid(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def ring_adjacency(n_depots: int) -> list[tuple[int, int]]:
    """Bidirectional ring; the five-depot class gets two additional chords
    (tex, "Carbon parameters and transfer network"). ASSUMPTION: the chords
    connect the two shortest non-adjacent diagonals (0,2) and (1,3); the tex
    names the count (two) but not which vertices they join."""
    if n_depots < 2:
        return []
    if n_depots == 2:
        # (d, (d+1) % n) degenerates for n=2: it lists the single physical
        # edge {0,1} twice, as (0,1) and (1,0). Return it once; the caller
        # (network.build_transfer_network) already creates both directions.
        return [(0, 1)]
    edges = [(d, (d + 1) % n_depots) for d in range(n_depots)]
    if n_depots == 5:
        edges += [(0, 2), (1, 3)]
    return edges


def place_depots(n_depots: int, rng: np.random.Generator) -> list[Depot]:
    cx = cy = REGION / 2
    radius = REGION * 0.35
    depots = []
    for d in range(n_depots):
        angle = 2 * math.pi * d / n_depots
        jitter_r = rng.uniform(-1, 1) * REGION * POLYGON_JITTER_FRAC
        jitter_a = rng.uniform(-1, 1) * (math.pi / max(n_depots, 1)) * POLYGON_ANGLE_JITTER
        x = cx + (radius + jitter_r) * math.cos(angle + jitter_a)
        y = cy + (radius + jitter_r) * math.sin(angle + jitter_a)
        depots.append(Depot(d, float(x), float(y), VEHICLE_CAPACITY, 0, 0.0))
    return depots


def generate_customers(depots: list[Depot], n_customers: int, boundary_share: float,
                        rng: np.random.Generator) -> list[Customer]:
    n_depots = len(depots)
    n_boundary = round(n_customers * boundary_share)
    n_core = n_customers - n_boundary

    radius = REGION * 0.35
    spacing = 2 * radius * math.sin(math.pi / max(2, n_depots))
    cluster_std = max(0.3, spacing * CLUSTER_STD_FRAC)

    customers: list[Customer] = []

    # Core customers: round-robin across depots for balanced cluster sizes.
    for i in range(n_core):
        d = depots[i % n_depots]
        x = float(np.clip(rng.normal(d.x, cluster_std), 0.0, REGION))
        y = float(np.clip(rng.normal(d.y, cluster_std), 0.0, REGION))
        customers.append(Customer(i, x, y, 0.0, 0.0, d.idx, (d.idx,)))

    # Boundary customers: near the midpoint of a ring-adjacent depot pair,
    # eligible for exactly that pair; home depot is whichever is nearer.
    pairs = ring_adjacency(n_depots) or [(0, 0)]
    for i in range(n_boundary):
        a, b = pairs[i % len(pairs)]
        da, db = depots[a], depots[b]
        mx, my = (da.x + db.x) / 2, (da.y + db.y) / 2
        x = float(np.clip(rng.normal(mx, cluster_std), 0.0, REGION))
        y = float(np.clip(rng.normal(my, cluster_std), 0.0, REGION))
        home = a if euclid(x, y, da.x, da.y) <= euclid(x, y, db.x, db.y) else b
        elig = tuple(sorted({a, b}))
        customers.append(Customer(n_core + i, x, y, 0.0, 0.0, home, elig))

    perm = rng.permutation(len(customers))
    customers = [customers[i] for i in perm]
    for new_idx, c in enumerate(customers):
        c.idx = new_idx
    return customers


def generate_fleet(depots: list[Depot], customers: list[Customer]) -> list[Depot]:
    """Size vehicle counts and route-duration limits from realised base
    demand, targeting TARGET_UTILIZATION. Duration is set generously (never
    the binding constraint by default) because the notation section assumes
    "every scenario is operationally feasible"; no suite in Table 1 varies
    duration as an experimental factor."""
    by_depot: dict[int, list[Customer]] = {}
    for c in customers:
        by_depot.setdefault(c.home_depot, []).append(c)
    for d in depots:
        members = by_depot.get(d.idx, [])
        total_demand = sum(c.base_demand for c in members)
        d.n_vehicles = max(1, math.ceil(total_demand / (d.capacity * TARGET_UTILIZATION))) if members else 1
        if members:
            farthest = max(euclid(d.x, d.y, c.x, c.y) for c in members)
        else:
            farthest = REGION
        # ASSUMPTION: generous multiple of a round trip to the farthest
        # member so duration is not the binding constraint by default.
        d.max_duration = 6.0 * (2 * farthest + 1.0)
    return depots

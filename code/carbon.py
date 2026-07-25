"""Reference emissions and the corporate carbon-budget parameterisation.

Reference emissions use the "deterministic home-depot angle-sweep
construction under base demand" named in the "Carbon parameters and transfer
network" paragraph of the appendix. The route-emission formula reproduces
E_d^omega from the deterministic-equivalent formulation (tex, around
eq. near "The load-dependent emission charged to depot d is..."):

    E = sum_over_arcs theta * d_ij * [rho0 * x_ij + (rhoQ - rho0)/Q * f_ij]

which for a single vehicle traversing a fixed sequence collapses to
theta * distance * (rho0 + (rhoQ - rho0) * load_fraction) per arc.

theta, rho0, rhoQ are left symbolic in the tex (no numeric value given);
THETA/RHO_EMPTY/RHO_FULL below are ASSUMPTION constants, reused unchanged
from the E5 public-benchmark adapter (gpt_test/build_cordeau_cbar_benchmark.py)
so the two data sources share one illustrative emission calibration.
"""
from __future__ import annotations

import math

THETA = 1.0
RHO_EMPTY = 0.9
RHO_FULL = 1.5

# tex, subsec:experimental_setup: "lower and upper responsibility bounds
# equal to 45% and 160% of that allocation."
BUDGET_LOWER_FRAC = 0.45
BUDGET_UPPER_FRAC = 1.60


def angle_sweep_routes(depot, members, capacity: float) -> list[list]:
    if not members:
        return []
    ordered = sorted(members, key=lambda c: math.atan2(c.y - depot.y, c.x - depot.x))
    routes, current, load = [], [], 0.0
    for c in ordered:
        if current and load + c.base_demand > capacity + 1e-9:
            routes.append(current)
            current, load = [], 0.0
        current.append(c)
        load += c.base_demand
    if current:
        routes.append(current)
    return routes


def route_emission(depot, route: list, capacity: float) -> float:
    if not route:
        return 0.0
    remaining = sum(c.base_demand for c in route)
    e = 0.0
    px, py = depot.x, depot.y
    for c in route:
        factor = RHO_EMPTY + (RHO_FULL - RHO_EMPTY) * max(0.0, min(capacity, remaining)) / capacity
        e += THETA * math.hypot(px - c.x, py - c.y) * factor
        remaining -= c.base_demand
        px, py = c.x, c.y
    factor = RHO_EMPTY + (RHO_FULL - RHO_EMPTY) * max(0.0, min(capacity, remaining)) / capacity
    e += THETA * math.hypot(px - depot.x, py - depot.y) * factor
    return e


def reference_emission_per_depot(depots, customers) -> dict:
    by_depot: dict[int, list] = {}
    for c in customers:
        by_depot.setdefault(c.home_depot, []).append(c)
    E0 = {}
    for d in depots:
        members = by_depot.get(d.idx, [])
        routes = angle_sweep_routes(d, members, d.capacity)
        E0[d.idx] = sum(route_emission(d, r, d.capacity) for r in routes)
    return E0


def corporate_budget(E0_by_depot: dict, gamma_B: float):
    """eq. (budget_stringency): B^corp = gamma_B * E^0, with the reference
    per-depot allocation proportional to E0 and box bounds at 45%/160% of
    that allocation (tex, subsec:experimental_setup)."""
    E0_total = sum(E0_by_depot.values())
    if E0_total <= 0:
        raise ValueError("total reference emission must be strictly positive")
    B_corp = gamma_B * E0_total
    alloc = {d: B_corp * e / E0_total for d, e in E0_by_depot.items()}
    bounds = {d: (BUDGET_LOWER_FRAC * a, BUDGET_UPPER_FRAC * a) for d, a in alloc.items()}
    return B_corp, alloc, bounds

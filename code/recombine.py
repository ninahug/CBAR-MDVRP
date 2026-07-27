"""Exact route recombination over a pool of generated routes.

At a fixed price vector the scenario objective

    sum_d [ distance_d + lambda_d * emission_d ]

is a sum over routes: a route contributes its own distance and its own
emission, priced at the depot it is rooted in, and no term couples two routes.
The best combination of any collection of feasible routes is therefore the
optimal solution of a set-partitioning problem, which is solved exactly.

This is the classical route-pool device of the vehicle-routing matheuristic
literature. What the present model adds is that the objective being recombined
is a linearisation whose accuracy is testable: if the emission vector of the
recombined plan leaves the deficit on the same piece of the recourse value
function, then by Proposition 8 the recombination is optimal for the exact
objective too, not only for the linearisation.

A neighbourhood search explores one trajectory and can only ever hold one
route per customer at a time. Recombination is not subject to that: it may
assemble a plan from routes discovered at different points of the search, and
in particular from routes rooted at different depots, which is where the
cross-depot reassignment specific to this problem is decided.
"""
from __future__ import annotations

import pulp

from instance_io import Instance
import ppbrc


class RoutePool:
    """Feasible routes seen during the search, keyed to remove duplicates.

    A route is stored with the depot it is rooted in, since the same customer
    sequence has a different cost and a different emission from a different
    depot, and the two are distinct columns of the set-partitioning problem.
    """

    def __init__(self, inst: Instance, demand: dict[int, float]):
        self.inst = inst
        self.demand = demand
        self._routes: dict[tuple, tuple] = {}

    def add_solution(self, sol: "ppbrc.Solution") -> None:
        for depot, routes in sol.routes.items():
            for route in routes:
                self.add(depot, route)

    def add(self, depot: int, route: list[int]) -> None:
        if not route:
            return
        key = (depot, tuple(route))
        if key in self._routes:
            return
        m = ppbrc.route_metrics(self.inst, depot, route, self.demand)
        if not m["feasible"]:
            return
        self._routes[key] = (m["distance"], m["emission"])

    def __len__(self) -> int:
        return len(self._routes)

    def columns(self):
        for (depot, route), (dist, emis) in self._routes.items():
            yield depot, list(route), dist, emis


def recombine(inst: Instance, demand: dict[int, float], pool: RoutePool,
               lam: dict[int, float], time_limit: float = 10.0):
    """Best combination of pooled routes under the priced objective.

    Returns a Solution, or None if the set-partitioning problem is not solved
    to optimality within the time limit. Every customer must be covered
    exactly once and no depot may exceed its fleet, so any returned plan is
    feasible for the scenario by construction.
    """
    customers = [c["idx"] for c in inst.customers]
    cols = list(pool.columns())
    if not cols:
        return None

    prob = pulp.LpProblem("route_recombination", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{k}", cat="Binary") for k in range(len(cols))]

    prob += pulp.lpSum(
        (dist + lam.get(depot, 0.0) * emis) * x[k]
        for k, (depot, route, dist, emis) in enumerate(cols)
    )

    covering = {i: [] for i in customers}
    by_depot: dict[int, list] = {d["idx"]: [] for d in inst.depots}
    for k, (depot, route, _dist, _emis) in enumerate(cols):
        by_depot[depot].append(x[k])
        for i in route:
            covering[i].append(x[k])

    for i in customers:
        if not covering[i]:
            return None          # a customer no pooled route serves
        prob += pulp.lpSum(covering[i]) == 1, f"cover_{i}"

    for d in inst.depots:
        if by_depot[d["idx"]]:
            prob += pulp.lpSum(by_depot[d["idx"]]) <= d["n_vehicles"], f"fleet_{d['idx']}"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    if pulp.LpStatus[status] != "Optimal":
        return None

    sol = ppbrc.Solution(inst, demand)
    for k, (depot, route, _dist, _emis) in enumerate(cols):
        if x[k].value() is not None and x[k].value() > 0.5:
            sol.routes[depot].append(list(route))
    if sum(len(r) for rs in sol.routes.values() for r in rs) != len(customers):
        return None
    return sol

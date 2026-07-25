"""Layer 3 of PPBRC: the exact budget master problem.

For fixed per-scenario emissions E^omega (routing/emissions frozen), the
budget-recourse problem

    min_{B in bounded simplex} sum_w pi_w Psi_Tbar(E^w - B)

is a single convex piecewise-linear (in fact linear-programming) problem
because Psi_Tbar is a min-cost-flow value (Prop. recourse_flow) -- so it can
be solved exactly by expanding the recourse LP once per scenario and sharing
one first-stage B across all of them, rather than approximated. This module
also solves the "wait-and-see" relaxation (an independent B per scenario) so
callers can compute VSS = Z*_stochastic - Z*_wait_and_see >= 0
(Section: "E3: value of stochastic carbon-budget allocation").
"""
from __future__ import annotations

import pulp


def _recourse_terms(prob, w, depots, transfer_arcs, p_buy, p_sell):
    arc_keys = [(a["from"], a["to"]) for a in transfer_arcs]
    cap = {(a["from"], a["to"]): a["capacity"] for a in transfer_arcs}
    kappa = {(a["from"], a["to"]): a["friction"] for a in transfer_arcs}
    t = {a: pulp.LpVariable(f"t_{a[0]}_{a[1]}_w{w}", lowBound=0, upBound=cap[a]) for a in arc_keys}
    b = {d: pulp.LpVariable(f"b_{d}_w{w}", lowBound=0) for d in depots}
    s = {d: pulp.LpVariable(f"s_{d}_w{w}", lowBound=0) for d in depots}
    cost = (pulp.lpSum(kappa[a] * t[a] for a in arc_keys)
            + p_buy * pulp.lpSum(b.values()) - p_sell * pulp.lpSum(s.values()))
    return t, b, s, arc_keys, cost


def solve_budget_master(E_by_scenario: dict[int, dict[int, float]], pi: dict[int, float],
                         depots: list[int], transfer_arcs: list[dict], prices: dict,
                         bounds: dict[int, tuple], corporate_budget: float,
                         per_scenario_B: bool = False) -> dict:
    """per_scenario_B=False: one shared B (the actual model). per_scenario_B=True:
    an independent B^w per scenario (wait-and-see relaxation, used only for VSS)."""
    p_buy, p_sell = prices["buy"], prices["sell"]
    prob = pulp.LpProblem("budget_master", pulp.LpMinimize)

    if per_scenario_B:
        B = {w: {d: pulp.LpVariable(f"B_{d}_w{w}", lowBound=bounds[d][0], upBound=bounds[d][1])
                 for d in depots} for w in E_by_scenario}
        for w in E_by_scenario:
            prob += pulp.lpSum(B[w][d] for d in depots) == corporate_budget, f"budget_total_w{w}"
    else:
        B_shared = {d: pulp.LpVariable(f"B_{d}", lowBound=bounds[d][0], upBound=bounds[d][1])
                    for d in depots}
        prob += pulp.lpSum(B_shared.values()) == corporate_budget, "budget_total"
        B = {w: B_shared for w in E_by_scenario}

    total_cost = 0
    for w, Ew in E_by_scenario.items():
        t, b, s, arc_keys, cost = _recourse_terms(prob, w, depots, transfer_arcs, p_buy, p_sell)
        for d in depots:
            incoming = pulp.lpSum(t[a] for a in arc_keys if a[1] == d)
            outgoing = pulp.lpSum(t[a] for a in arc_keys if a[0] == d)
            prob += (B[w][d] + incoming + b[d] == Ew[d] + outgoing + s[d], f"balance_{d}_w{w}")
        total_cost += pi[w] * cost

    prob += total_cost
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return {
        "status": pulp.LpStatus[status],
        "value": pulp.value(prob.objective),
        "B": ({d: B[next(iter(B))][d].value() for d in depots} if not per_scenario_B
              else {w: {d: B[w][d].value() for d in depots} for w in E_by_scenario}),
    }


def value_of_stochastic_solution(E_by_scenario, pi, depots, transfer_arcs, prices, bounds, corporate_budget) -> dict:
    stochastic = solve_budget_master(E_by_scenario, pi, depots, transfer_arcs, prices, bounds,
                                      corporate_budget, per_scenario_B=False)
    wait_and_see = solve_budget_master(E_by_scenario, pi, depots, transfer_arcs, prices, bounds,
                                        corporate_budget, per_scenario_B=True)
    vss = stochastic["value"] - wait_and_see["value"]
    return {"stochastic": stochastic["value"], "wait_and_see": wait_and_see["value"], "vss": vss}

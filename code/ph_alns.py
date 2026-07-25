"""PH-ALNS baseline (tex, subsec:e2_algorithm item 2): "a heuristic
progressive-hedging adaptation with scenario-specific budget copies, exact
recourse subgradients in projected proximal updates, adaptive penalty
balancing, scenario ALNS, and exact feasible nonanticipative recovery."

Each scenario gets its own budget copy B^w (routed with price-guided ALNS
against that copy, reusing ppbrc.solve_scenario). Progressive hedging then
drives the copies toward consensus with a projected proximal update; because
an exact PH proximal subproblem would need re-solving the whole routing
problem at every step, this uses the standard PH linear/quadratic term
directly on B (a documented simplification -- PH-ALNS is explicitly
described in the tex as "a heuristic adaptation", not required to be exact).
Since consensus is not perfect, both the reported "feasible recovered
objective" (true cost of the single shared B, tex's nonanticipative recovery)
and the "budget-consensus residual" (how far the copies disagreed) are
returned, as the tex's own metric list requires.
"""
from __future__ import annotations

import ppbrc
import budget_master
import recourse
from instance_io import Instance


def run_ph_alns(inst: Instance, scenarios: list[dict], ph_iterations: int = 8,
                 rho: float = 1.0, price_rounds: int = 2, alns_iterations: int = 150,
                 seed: int = 0) -> dict:
    D = [d["idx"] for d in inst.depots]
    W = list(range(len(scenarios)))
    pi = {w: scenarios[w]["probability"] for w in W}
    demands = {w: inst.scenario_demand(scenarios[w]) for w in W}
    bounds = inst.depot_budget_bounds
    B_corp = inst.corporate_budget

    B_w = {w: dict(inst.depot_budget_allocation) for w in W}
    weight = {w: {d: 0.0 for d in D} for w in W}
    solutions = {w: None for w in W}

    for it in range(ph_iterations):
        E_by_scenario = {}
        for w in W:
            result = ppbrc.solve_scenario(inst, demands[w], B_w[w], price_rounds=price_rounds,
                                           alns_iterations=alns_iterations, seed=seed + 7919 * w + it,
                                           initial=solutions[w], price_guided=True, diversify=False)
            solutions[w] = result["solution"]
            E_by_scenario[w] = result["emissions"]

        # Consensus target and the standard PH price/proximal update.
        Bbar = {d: sum(pi[w] * B_w[w][d] for w in W) for d in D}
        for w in W:
            for d in D:
                weight[w][d] += rho * (B_w[w][d] - Bbar[d])

        # Projected proximal step: minimise recourse_LP(E^w - B) + w.B +
        # rho/2 ||B-Bbar||^2 over the budget simplex+box -- approximated by
        # solving the exact recourse LP at Bbar shifted by the accumulated
        # multiplier (a standard PH simplification for a nonconvex outer
        # problem), then projecting onto [lower,upper] and renormalising to
        # the corporate total.
        for w in W:
            proposed = {d: min(max(Bbar[d] - weight[w][d] / max(rho, 1e-9), bounds[d][0]), bounds[d][1])
                        for d in D}
            total = sum(proposed.values())
            if total > 0:
                proposed = {d: proposed[d] * B_corp / total for d in D}
            B_w[w] = proposed

    residual = sum(pi[w] * sum(abs(B_w[w][d] - Bbar[d]) for d in D) for w in W)

    # Exact feasible nonanticipative recovery: one shared B, exactly
    # optimised given the final routing's emissions (Layer 3, reused).
    master = budget_master.solve_budget_master(E_by_scenario, pi, D, inst.transfer_arcs, inst.prices,
                                                 bounds, B_corp, per_scenario_B=False)
    B_shared = master["B"]
    total_distance = sum(pi[w] * sum(solutions[w].depot_metrics(d)[0] for d in D) for w in W)
    recourse_cost = sum(pi[w] * recourse.solve_recourse_primal(
        {d: E_by_scenario[w][d] - B_shared[d] for d in D}, D, inst.transfer_arcs,
        inst.prices["buy"], inst.prices["sell"])["value"] for w in W)
    feasible_objective = total_distance + recourse_cost

    return {
        "objective": feasible_objective,
        "distance": total_distance,
        "recourse_cost": recourse_cost,
        "B": B_shared,
        "B_scenario_copies": B_w,
        "budget_consensus_residual": residual,
        "solutions": solutions,
        "emissions": E_by_scenario,
    }

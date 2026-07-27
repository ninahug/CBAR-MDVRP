"""Carbon-recourse value function Psi_Tbar(g) and its price-polyhedron dual.

Implements Section "Carbon-Recourse Value and Price-Polyhedral Structure":
  - primal min-cost-flow LP (eq. recourse_primal/recourse_balance/
    recourse_capacity/recourse_domain)
  - dual LP over the extended price polyhedron (eq. recourse_dual/
    dual_price_bounds/dual_transfer_arbitrage/dual_mu_domain)

Both are solved independently (not by reading solver-internal dual values off
the primal) so that a primal-dual mismatch is a real, independent numerical
check of Prop. price_polyhedron -- exactly the check the original design doc
(logical-splashing-pixel.md) flagged as "claimed in prose, never actually
run in code" for the previous model version.
"""
from __future__ import annotations

import pulp


def solve_recourse_primal(g: dict[int, float], depots: list[int], transfer_arcs: list[dict],
                           p_buy: float, p_sell: float, canonical: bool = True,
                           tol: float = 1e-7):
    """Minimum-cost carbon-account recourse at deficit `g`.

    A minimum-cost flow need not have a unique optimal solution: an optimal
    flow may be augmented around a zero-reduced-cost cycle without changing
    the cost. Reported internal-transfer volumes would then depend on which
    optimum the solver happened to return. With canonical=True a second stage
    holds the cost at its optimum and minimises total internal transfer over
    that optimal face, which selects a well-defined representative and removes
    gratuitous circulation while leaving Psi_Tbar(g) unchanged. This is the
    rule the paper states in Section 5.2 and Appendix A.
    """
    def build():
        prob = pulp.LpProblem("recourse_primal", pulp.LpMinimize)
        arc_keys = [(a["from"], a["to"]) for a in transfer_arcs]
        cap = {(a["from"], a["to"]): a["capacity"] for a in transfer_arcs}
        kappa = {(a["from"], a["to"]): a["friction"] for a in transfer_arcs}
        t = {a: pulp.LpVariable(f"t_{a[0]}_{a[1]}", lowBound=0, upBound=cap[a]) for a in arc_keys}
        b = {d: pulp.LpVariable(f"b_{d}", lowBound=0) for d in depots}
        s = {d: pulp.LpVariable(f"s_{d}", lowBound=0) for d in depots}
        cost = (pulp.lpSum(kappa[a] * t[a] for a in arc_keys)
                + p_buy * pulp.lpSum(b.values())
                - p_sell * pulp.lpSum(s.values()))
        for d in depots:
            incoming = pulp.lpSum(t[a] for a in arc_keys if a[1] == d)
            outgoing = pulp.lpSum(t[a] for a in arc_keys if a[0] == d)
            prob += (b[d] + incoming - outgoing - s[d] == g[d]), f"balance_{d}"
        return prob, t, b, s, arc_keys, cost

    prob, t, b, s, arc_keys, cost = build()
    prob += cost
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    value = pulp.value(prob.objective)

    if canonical and arc_keys and value is not None:
        prob2, t2, b2, s2, arc_keys2, cost2 = build()
        prob2 += pulp.lpSum(t2.values())
        prob2 += (cost2 <= value + tol), "hold_optimal_cost"
        status2 = prob2.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status2] == "Optimal":
            t, b, s = t2, b2, s2
            status = status2
    return {
        "status": pulp.LpStatus[status],
        "value": value,
        "t": {a: t[a].value() for a in arc_keys},
        "b": {d: b[d].value() for d in depots},
        "s": {d: s[d].value() for d in depots},
    }


def solve_recourse_dual(g: dict[int, float], depots: list[int], transfer_arcs: list[dict],
                         p_buy: float, p_sell: float):
    prob = pulp.LpProblem("recourse_dual", pulp.LpMaximize)
    arc_keys = [(a["from"], a["to"]) for a in transfer_arcs]
    cap = {(a["from"], a["to"]): a["capacity"] for a in transfer_arcs}
    kappa = {(a["from"], a["to"]): a["friction"] for a in transfer_arcs}

    lam = {d: pulp.LpVariable(f"lambda_{d}", lowBound=p_sell, upBound=p_buy) for d in depots}
    mu = {a: pulp.LpVariable(f"mu_{a[0]}_{a[1]}", lowBound=0) for a in arc_keys}

    prob += (pulp.lpSum(lam[d] * g[d] for d in depots)
             - pulp.lpSum(cap[a] * mu[a] for a in arc_keys))

    for a in arc_keys:
        d, dp = a
        prob += lam[dp] - lam[d] - mu[a] <= kappa[a], f"arbitrage_{d}_{dp}"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    value = pulp.value(prob.objective)
    return {
        "status": pulp.LpStatus[status],
        "value": value,
        "lambda": {d: lam[d].value() for d in depots},
        "mu": {a: mu[a].value() for a in arc_keys},
    }


def solve_recourse(g: dict[int, float], depots: list[int], transfer_arcs: list[dict],
                    prices: dict) -> dict:
    p_buy, p_sell = prices["buy"], prices["sell"]
    primal = solve_recourse_primal(g, depots, transfer_arcs, p_buy, p_sell)
    dual = solve_recourse_dual(g, depots, transfer_arcs, p_buy, p_sell)
    duality_gap = abs(primal["value"] - dual["value"])
    return {
        "primal": primal,
        "dual": dual,
        "value": primal["value"],
        "duality_gap": duality_gap,
    }


def check_complementarity(primal: dict, dual: dict, transfer_arcs: list[dict],
                           p_buy: float, p_sell: float, tol: float = 1e-6) -> list[str]:
    """Prop. complementarity, checks 1-5."""
    errors = []
    lam = dual["lambda"]
    for d, val in primal["b"].items():
        if val > tol and abs(lam[d] - p_buy) > tol:
            errors.append(f"depot {d}: b_d={val:.4f}>0 but lambda_d={lam[d]:.4f} != p_buy={p_buy}")
    for d, val in primal["s"].items():
        if val > tol and abs(lam[d] - p_sell) > tol:
            errors.append(f"depot {d}: s_d={val:.4f}>0 but lambda_d={lam[d]:.4f} != p_sell={p_sell}")
    for a in transfer_arcs:
        key = (a["from"], a["to"])
        t_val = primal["t"][key]
        d, dp = key
        lhs = lam[dp] - lam[d]
        if t_val < tol:
            if lhs - a["friction"] > tol:
                errors.append(f"arc {key}: unused but lambda diff {lhs:.4f} > friction {a['friction']:.4f}")
        elif t_val < a["capacity"] - tol:
            if abs(lhs - a["friction"]) > tol:
                errors.append(f"arc {key}: interior flow but lambda diff {lhs:.4f} != friction {a['friction']:.4f}")
        else:
            if lhs - a["friction"] < -tol:
                errors.append(f"arc {key}: saturated but lambda diff {lhs:.4f} < friction {a['friction']:.4f}")
    return errors

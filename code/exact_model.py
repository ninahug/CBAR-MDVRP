"""Deterministic-equivalent MILP (tex Section 3, eq. obj / c1-c20 / budget_*
/ account_balance / transfer_capacity / vi1-vi3), translated to PuLP/CBC.

Node convention: a depot's origin and terminal are distinct string nodes
"O{d}"/"R{d}" (shared by every vehicle stationed at depot d, per the tex's
o_k = o_{d(k)}, r_k = r_{d(k)}); customers are plain integer indices.

Defaults not produced by the data generator (see instance_io.py docstring):
dispatch cost F_k = 0, travel cost/time = Euclidean distance at unit speed.
"""
from __future__ import annotations

import math
import re
import tempfile
import time
from pathlib import Path

import pulp

from instance_io import Instance


def onode(depot: int) -> str:
    return f"O{depot}"


def rnode(depot: int) -> str:
    return f"R{depot}"


def node_xy(inst: Instance, node):
    if isinstance(node, str):
        depot = int(node[1:])
        return inst.depot_xy(depot)
    return inst.customer_xy(node)


def dist(inst: Instance, u, v) -> float:
    ax, ay = node_xy(inst, u)
    bx, by = node_xy(inst, v)
    return math.hypot(ax - bx, ay - by)


def arcs_of_vehicle(inst: Instance, k: int) -> list[tuple]:
    depot = inst.vehicles[k].depot
    cks = inst.customers_of_vehicle[k]
    o, r = onode(depot), rnode(depot)
    arcs = [(o, j) for j in cks]
    arcs += [(i, j) for i in cks for j in cks if i != j]
    arcs += [(i, r) for i in cks]
    return arcs


def build_and_solve(inst: Instance, scenarios: list[dict], time_limit: float = 120.0,
                     msg: bool = False, use_valid_inequalities: bool = True, gap_rel: float = 1e-9):
    D = [d["idx"] for d in inst.depots]
    C = [c["idx"] for c in inst.customers]
    K = list(range(len(inst.vehicles)))
    W = list(range(len(scenarios)))
    pi = {w: scenarios[w]["probability"] for w in W}
    q = {w: inst.scenario_demand(scenarios[w]) for w in W}

    vehicles_of_depot = {d: [k for k in K if inst.vehicles[k].depot == d] for d in D}
    arcs_k = {k: arcs_of_vehicle(inst, k) for k in K}
    Ck = inst.customers_of_vehicle
    F_k = 0.0  # ASSUMPTION: no per-vehicle dispatch cost is generated

    prob = pulp.LpProblem("cbar_mdvrp_exact", pulp.LpMinimize)

    y = {(i, d, w): pulp.LpVariable(f"y_{i}_{d}_{w}", cat="Binary")
         for i in C for d in inst.eligible[i] for w in W}
    z = {(i, k, w): pulp.LpVariable(f"z_{i}_{k}_{w}", cat="Binary")
         for k in K for i in Ck[k] for w in W}
    a = {(k, w): pulp.LpVariable(f"a_{k}_{w}", cat="Binary") for k in K for w in W}
    x = {(u, v, k, w): pulp.LpVariable(f"x_{u}_{v}_{k}_{w}", cat="Binary")
         for k in K for (u, v) in arcs_k[k] for w in W}
    f = {(u, v, k, w): pulp.LpVariable(f"f_{u}_{v}_{k}_{w}", lowBound=0)
         for k in K for (u, v) in arcs_k[k] for w in W}
    E = {(d, w): pulp.LpVariable(f"E_{d}_{w}", lowBound=0) for d in D for w in W}

    arc_keys = [(ta["from"], ta["to"]) for ta in inst.transfer_arcs]
    cap = {(ta["from"], ta["to"]): ta["capacity"] for ta in inst.transfer_arcs}
    kappa = {(ta["from"], ta["to"]): ta["friction"] for ta in inst.transfer_arcs}
    t = {(u, v, w): pulp.LpVariable(f"t_{u}_{v}_{w}", lowBound=0, upBound=cap[(u, v)])
         for (u, v) in arc_keys for w in W}
    b = {(d, w): pulp.LpVariable(f"b_{d}_{w}", lowBound=0) for d in D for w in W}
    s = {(d, w): pulp.LpVariable(f"s_{d}_{w}", lowBound=0) for d in D for w in W}

    B = {d: pulp.LpVariable(f"B_{d}", lowBound=inst.depot_budget_bounds[d][0],
                             upBound=inst.depot_budget_bounds[d][1]) for d in D}

    p_buy, p_sell = inst.prices["buy"], inst.prices["sell"]

    prob += pulp.lpSum(
        pi[w] * (
            pulp.lpSum(F_k * a[k, w] for k in K)
            + pulp.lpSum(dist(inst, u, v) * x[u, v, k, w] for k in K for (u, v) in arcs_k[k])
            + inst.emission_cost * pulp.lpSum(
                inst.theta * dist(inst, u, v) * (
                    inst.rho_empty * x[u, v, k, w]
                    + (inst.rho_full - inst.rho_empty) / inst.vehicles[k].capacity * f[u, v, k, w])
                for k in K for (u, v) in arcs_k[k])
            + pulp.lpSum(kappa[uv] * t[uv[0], uv[1], w] for uv in arc_keys)
            + p_buy * pulp.lpSum(b[d, w] for d in D)
            - p_sell * pulp.lpSum(s[d, w] for d in D)
        )
        for w in W
    )

    # eq. budget_total / budget_bounds (bounds already on the B[d] variable)
    prob += pulp.lpSum(B[d] for d in D) == inst.corporate_budget, "budget_total"

    for w in W:
        qw = q[w]
        # c1
        for i in C:
            prob += pulp.lpSum(y[i, d, w] for d in inst.eligible[i]) == 1, f"c1_{i}_{w}"
        # c2
        for i in C:
            for d in inst.eligible[i]:
                prob += (pulp.lpSum(z[i, k, w] for k in vehicles_of_depot[d]) == y[i, d, w],
                         f"c2_{i}_{d}_{w}")
        for k in K:
            # c3, c4
            for i in Ck[k]:
                prob += z[i, k, w] <= a[k, w], f"c3_{i}_{k}_{w}"
            prob += pulp.lpSum(z[i, k, w] for i in Ck[k]) >= a[k, w], f"c4_{k}_{w}"
            depot = inst.vehicles[k].depot
            o, r = onode(depot), rnode(depot)
            # c56
            prob += (pulp.lpSum(x[o, j, k, w] for j in Ck[k]) == a[k, w], f"c5_{k}_{w}")
            prob += (pulp.lpSum(x[i, r, k, w] for i in Ck[k]) == a[k, w], f"c6_{k}_{w}")
            # c7 (out-degree), c78 (in-degree)
            for i in Ck[k]:
                out_nodes = [j for j in Ck[k] if j != i] + [r]
                in_nodes = [j for j in Ck[k] if j != i] + [o]
                prob += (pulp.lpSum(x[i, j, k, w] for j in out_nodes) == z[i, k, w],
                         f"c7_{i}_{k}_{w}")
                prob += (pulp.lpSum(x[j, i, k, w] for j in in_nodes) == z[i, k, w],
                         f"c78_{i}_{k}_{w}")
            # c12 (duration; service time = 0)
            prob += (pulp.lpSum(dist(inst, u, v) * x[u, v, k, w] for (u, v) in arcs_k[k])
                     <= inst.vehicles[k].max_duration * a[k, w], f"c12_{k}_{w}")
            # c16
            prob += (pulp.lpSum(f[o, j, k, w] for j in Ck[k])
                     == pulp.lpSum(qw[i] * z[i, k, w] for i in Ck[k]), f"c16_{k}_{w}")
            # c17
            for i in Ck[k]:
                out_nodes = [j for j in Ck[k] if j != i] + [r]
                in_nodes = [j for j in Ck[k] if j != i] + [o]
                prob += (pulp.lpSum(f[j, i, k, w] for j in in_nodes)
                         - pulp.lpSum(f[i, j, k, w] for j in out_nodes)
                         == qw[i] * z[i, k, w], f"c17_{i}_{k}_{w}")
            # c19a/c19b + valid inequalities vi1/vi2
            for (u, v) in arcs_k[k]:
                prob += f[u, v, k, w] <= inst.vehicles[k].capacity * x[u, v, k, w], f"c19a_{u}_{v}_{k}_{w}"
                if use_valid_inequalities and isinstance(v, int):
                    prob += f[u, v, k, w] >= qw[v] * x[u, v, k, w], f"vi1_{u}_{v}_{k}_{w}"
                if use_valid_inequalities and isinstance(u, int):
                    prob += (f[u, v, k, w] <= (inst.vehicles[k].capacity - qw[u]) * x[u, v, k, w],
                             f"vi2_{u}_{v}_{k}_{w}")
            for i in Ck[k]:
                prob += f[i, r, k, w] == 0, f"c19b_{i}_{k}_{w}"

        # c20: the stock consumed at depot d is the demand assigned to it.
        for d in D:
            prob += (E[d, w] == pulp.lpSum(qw[i] * y[i, d, w]
                                            for i in C if d in inst.eligible[i]),
                     f"c20_{d}_{w}")

        # account_balance
        for d in D:
            incoming = pulp.lpSum(t[h, d, w] for (h, dd) in arc_keys if dd == d)
            outgoing = pulp.lpSum(t[d, h, w] for (dd, h) in arc_keys if dd == d)
            prob += (B[d] + incoming + b[d, w] == E[d, w] + outgoing + s[d, w],
                     f"balance_{d}_{w}")

        # vi3: symmetry breaking across identical vehicles at the same depot
        if use_valid_inequalities:
            for d in D:
                ks = vehicles_of_depot[d]
                for k1, k2 in zip(ks, ks[1:]):
                    prob += a[k1, w] >= a[k2, w], f"vi3_{k1}_{k2}_{w}"

    log_path = Path(tempfile.gettempdir()) / f"cbar_cbc_{id(prob)}.log"
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, logPath=str(log_path), gapRel=gap_rel)
    t0 = time.time()
    status = prob.solve(solver)
    runtime = time.time() - t0

    # CBC/PuLP will report an "Optimal" status and a variable assignment even
    # when no integer-feasible incumbent was found in time -- in that case
    # the returned x/y/z/a values are the last (fractional) LP relaxation
    # point, not a real routing plan, and pulp.value(prob.objective) is
    # meaningless as an incumbent. Checked explicitly here rather than
    # trusted, because a fractional "solution" was silently compared against
    # PPBRC's real feasible solutions in an earlier run and produced
    # nonsensical gaps (including an apparently "proven" optimum that was
    # actually just a reproducible fractional relaxation point).
    binary_vars = list(x.values()) + list(y.values()) + list(z.values()) + list(a.values())
    max_frac = 0.0
    for v in binary_vars:
        val = v.value()
        if val is not None:
            max_frac = max(max_frac, min(val, 1 - val) if 0 <= val <= 1 else abs(round(val) - val))
    integer_feasible = max_frac < 1e-4

    bound = pulp.value(prob.objective)
    proven_optimal = False
    try:
        log_text = log_path.read_text(errors="replace")
        m = re.search(r"Lower bound:\s*(-?[\d.]+)", log_text)
        if m:
            bound = float(m.group(1))
        proven_optimal = integer_feasible and "Stopped on time limit" not in log_text and bound is not None
    except OSError:
        pass

    return {
        "status": pulp.LpStatus[status],
        "objective": pulp.value(prob.objective) if integer_feasible else None,
        "bound": bound,
        "integer_feasible": integer_feasible,
        "max_fractionality": max_frac,
        "proven_optimal": proven_optimal,
        "runtime": runtime,
        "vars": {"y": y, "z": z, "a": a, "x": x, "f": f, "E": E, "t": t, "b": b, "s": s, "B": B},
    }

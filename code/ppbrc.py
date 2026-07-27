"""PPBRC: Price-Polyhedral Budget-Routing Coordination (tex Section 4).

Three layers, matching the design doc (logical-splashing-pixel.md) and the
tex's Section 4 / Appendix "PPBRC implementation details" almost verbatim:

  Layer 1 (this module, `alns_solve`): given a fixed per-depot price vector
    lambda^omega, solve min C_op(S^omega) + sum_d lambda_d^omega E_d(S^omega)
    with an ALNS (random/worst/Shaw/boundary removal, greedy/regret-2
    repair, simulated-annealing acceptance). Because lambda is fixed for the
    whole ALNS run, move evaluation needs no inner LP call -- it is a
    linearly-priced VRP. Route emission/distance/duration are recomputed by
    a full O(route length) walk per candidate move; at the instance sizes
    here (<=360 customers) this is fast enough that the incremental O(1)
    caching from the original fast_engine design isn't needed for
    correctness-first Stage 1/2 -- can be added later purely for speed if
    profiling shows it's the bottleneck.

  Layer 2 (`refit_prices`): given a routing solution's emissions E^omega,
    resolve the recourse dual (recourse.py) to get the new lambda^omega,
    mu^omega; fixed-point iterate with Layer 1 until the price stabilises or
    a round cap is hit.

  Layer 3: the exact budget master (budget_master.py, already built in
  Stage 0) -- not duplicated here.

`run_ppbrc` drives the outer coordination loop across all scenarios and the
shared budget B.
"""
from __future__ import annotations

import math
import random

from instance_io import Instance
import carbon
import recourse
import budget_master

THETA = carbon.THETA
RHO_EMPTY = carbon.RHO_EMPTY
RHO_FULL = carbon.RHO_FULL


def route_metrics(inst: Instance, depot: int, seq: list[int], demand: dict[int, float]) -> dict:
    """Full recompute of one route's distance, load-dependent emission, and
    duration (unit-speed travel time == distance, service time == 0)."""
    dep = inst.depots[depot]
    capacity = dep["capacity"]
    if not seq:
        return {"distance": 0.0, "emission": 0.0, "duration": 0.0, "load": 0.0, "feasible": True}
    load = sum(demand[i] for i in seq)
    remaining = load
    distance = 0.0
    emission = 0.0
    px, py = dep["x"], dep["y"]
    for i in seq:
        c = inst.customers[i]
        d = math.hypot(px - c["x"], py - c["y"])
        distance += d
        factor = RHO_EMPTY + (RHO_FULL - RHO_EMPTY) * max(0.0, min(capacity, remaining)) / capacity
        emission += THETA * d * factor
        remaining -= demand[i]
        px, py = c["x"], c["y"]
    d = math.hypot(px - dep["x"], py - dep["y"])
    distance += d
    factor = RHO_EMPTY + (RHO_FULL - RHO_EMPTY) * max(0.0, min(capacity, remaining)) / capacity
    emission += THETA * d * factor
    feasible = load <= capacity + 1e-6 and distance <= dep["max_duration"] + 1e-6
    return {"distance": distance, "emission": emission, "duration": distance, "load": load, "feasible": feasible}


class Solution:
    """routes[depot] = list of routes, each route a list of customer indices."""

    def __init__(self, inst: Instance, demand: dict[int, float]):
        self.inst = inst
        self.demand = demand
        self.routes: dict[int, list[list[int]]] = {d["idx"]: [] for d in inst.depots}
        self.unassigned: list[int] = []

    def clone(self) -> "Solution":
        s = Solution(self.inst, self.demand)
        s.routes = {d: [list(r) for r in rs] for d, rs in self.routes.items()}
        s.unassigned = list(self.unassigned)
        return s

    def depot_metrics(self, depot: int) -> tuple[float, float]:
        dist = emis = 0.0
        for r in self.routes[depot]:
            m = route_metrics(self.inst, depot, r, self.demand)
            dist += m["distance"]
            emis += m["emission"]
        return dist, emis

    def objective(self, lam: dict[int, float]) -> float:
        total = 0.0
        for d in self.routes:
            dist, emis = self.depot_metrics(d)
            total += dist + lam.get(d, 0.0) * emis
        total += len(self.unassigned) * 1e6  # heavy penalty: every customer must be served
        return total

    def emissions(self) -> dict[int, float]:
        return {d: self.depot_metrics(d)[1] for d in self.routes}


def _greedy_initial(inst: Instance, demand: dict[int, float], rng: random.Random) -> Solution:
    sol = Solution(inst, demand)
    order = list(demand.keys())
    rng.shuffle(order)
    for i in order:
        best = None
        for d in inst.eligible[i]:
            for ridx, r in enumerate(sol.routes[d]):
                for pos in range(len(r) + 1):
                    cand = r[:pos] + [i] + r[pos:]
                    m = route_metrics(inst, d, cand, demand)
                    if not m["feasible"]:
                        continue
                    old = route_metrics(inst, d, r, demand)
                    delta = m["distance"] - old["distance"]
                    if best is None or delta < best[0]:
                        best = (delta, d, ridx, pos)
            # option: open a new route at depot d
            cand = [i]
            m = route_metrics(inst, d, cand, demand)
            if m["feasible"] and len(sol.routes[d]) < _max_routes(inst, d):
                delta = m["distance"]
                if best is None or delta < best[0]:
                    best = (delta, d, None, 0)
        if best is None:
            sol.unassigned.append(i)
        else:
            _, d, ridx, pos = best
            if ridx is None:
                sol.routes[d].append([i])
            else:
                sol.routes[d][ridx].insert(pos, i)
    return sol


def _max_routes(inst: Instance, depot: int) -> int:
    return inst.depots[depot]["n_vehicles"]


def _remove_random(sol: Solution, k: int, rng: random.Random) -> list[int]:
    all_customers = [(d, ridx, pos, c) for d, rs in sol.routes.items()
                      for ridx, r in enumerate(rs) for pos, c in enumerate(r)]
    if not all_customers:
        return []
    picks = rng.sample(all_customers, min(k, len(all_customers)))
    return _apply_removal(sol, picks)


def _remove_worst(sol: Solution, k: int, lam: dict[int, float], rng: random.Random) -> list[int]:
    scored = []
    for d, rs in sol.routes.items():
        for ridx, r in enumerate(rs):
            base = route_metrics(sol.inst, d, r, sol.demand)
            for pos, c in enumerate(r):
                trial = r[:pos] + r[pos + 1:]
                m = route_metrics(sol.inst, d, trial, sol.demand)
                gain = (base["distance"] + lam.get(d, 0.0) * base["emission"]) - \
                       (m["distance"] + lam.get(d, 0.0) * m["emission"])
                scored.append((gain, d, ridx, pos, c))
    scored.sort(key=lambda t: -t[0])
    top = scored[:max(k * 3, k)]
    rng.shuffle(top)
    picks = [(d, ridx, pos, c) for _, d, ridx, pos, c in top[:k]]
    return _apply_removal(sol, picks)


def _remove_related(sol: Solution, k: int, rng: random.Random) -> list[int]:
    all_customers = [(d, ridx, pos, c) for d, rs in sol.routes.items()
                      for ridx, r in enumerate(rs) for pos, c in enumerate(r)]
    if not all_customers:
        return []
    seed_d, seed_ridx, seed_pos, seed_c = rng.choice(all_customers)
    cx, cy = sol.inst.customer_xy(seed_c)

    def dist_to_seed(item):
        _, _, _, c = item
        x, y = sol.inst.customer_xy(c)
        return math.hypot(x - cx, y - cy)

    all_customers.sort(key=dist_to_seed)
    picks = all_customers[:k]
    return _apply_removal(sol, picks)


def _remove_boundary(sol: Solution, k: int, rng: random.Random) -> list[int]:
    """Removes currently-routed boundary-eligible customers (|eligible|>=2)
    so repair can re-evaluate which of their eligible depots is actually
    better under the current price -- without this, a customer that greedy
    repair once assigned to depot A rarely gets re-tested against depot B,
    since random/worst/related removal touch it only by chance."""
    boundary = [(d, ridx, pos, c) for d, rs in sol.routes.items()
                for ridx, r in enumerate(rs) for pos, c in enumerate(r)
                if len(sol.inst.eligible[c]) >= 2]
    if not boundary:
        return []
    picks = rng.sample(boundary, min(k, len(boundary)))
    return _apply_removal(sol, picks)


def _apply_removal(sol: Solution, picks: list[tuple]) -> list[int]:
    removed = []
    by_route: dict[tuple, list[int]] = {}
    for d, ridx, pos, c in picks:
        by_route.setdefault((d, ridx), []).append(pos)
    for (d, ridx), positions in by_route.items():
        r = sol.routes[d][ridx]
        keep_positions = set(range(len(r))) - set(positions)
        removed.extend(r[pos] for pos in sorted(positions))
        sol.routes[d][ridx] = [r[pos] for pos in sorted(keep_positions)]
    for d in sol.routes:
        sol.routes[d] = [r for r in sol.routes[d] if r]
    sol.unassigned.extend(removed)
    return removed


def _insertion_candidates(sol: Solution, i: int, lam: dict[int, float]) -> list[tuple]:
    """All feasible (delta_cost, depot, route_index_or_None, position) for
    inserting customer i, sorted cheapest first. route_index None means
    "open a new route at that depot"."""
    cands = []
    for d in sol.inst.eligible[i]:
        for ridx, r in enumerate(sol.routes[d]):
            old = route_metrics(sol.inst, d, r, sol.demand)
            for pos in range(len(r) + 1):
                cand = r[:pos] + [i] + r[pos:]
                m = route_metrics(sol.inst, d, cand, sol.demand)
                if not m["feasible"]:
                    continue
                delta = (m["distance"] + lam.get(d, 0.0) * m["emission"]) - \
                        (old["distance"] + lam.get(d, 0.0) * old["emission"])
                cands.append((delta, d, ridx, pos))
        if len(sol.routes[d]) < _max_routes(sol.inst, d):
            m = route_metrics(sol.inst, d, [i], sol.demand)
            if m["feasible"]:
                cands.append((m["distance"] + lam.get(d, 0.0) * m["emission"], d, None, 0))
    cands.sort(key=lambda c: c[0])
    return cands


def _apply_insertion(sol: Solution, i: int, choice: tuple) -> None:
    _, d, ridx, pos = choice
    if ridx is None:
        sol.routes[d].append([i])
    else:
        sol.routes[d][ridx].insert(pos, i)


def _repair_greedy(sol: Solution, lam: dict[int, float], rng: random.Random) -> None:
    pending = list(sol.unassigned)
    sol.unassigned = []
    rng.shuffle(pending)
    while pending:
        best = None
        best_i = None
        for i in pending:
            cands = _insertion_candidates(sol, i, lam)
            if cands and (best is None or cands[0][0] < best[0]):
                best = cands[0]
                best_i = i
        if best is None:
            break
        _apply_insertion(sol, best_i, best)
        pending.remove(best_i)
    sol.unassigned.extend(pending)


def _repair_regret2(sol: Solution, lam: dict[int, float], rng: random.Random) -> None:
    """tex (E2, SAA-ALNS description): "greedy and regret-2 repair". Inserts
    the customer whose best-vs-second-best insertion cost gap is largest
    first -- this is what actually forces boundary customers to be tested
    against both eligible depots before the cheaper-looking one locks in,
    which plain greedy repair (evaluated one customer at a time, in
    shuffled order) does not reliably do."""
    pending = list(sol.unassigned)
    sol.unassigned = []
    while pending:
        best_regret = None
        best_i = None
        best_choice = None
        for i in pending:
            cands = _insertion_candidates(sol, i, lam)
            if not cands:
                continue
            regret = (cands[1][0] - cands[0][0]) if len(cands) > 1 else cands[0][0]
            if best_regret is None or regret > best_regret:
                best_regret = regret
                best_i = i
                best_choice = cands[0]
        if best_i is None:
            break
        _apply_insertion(sol, best_i, best_choice)
        pending.remove(best_i)
    sol.unassigned.extend(pending)


DESTROY_OPS = ["random", "worst", "related", "boundary"]
REPAIR_OPS = ["greedy", "regret2"]


def _multi_start_initial(inst: Instance, demand: dict[int, float], lam: dict[int, float],
                          rng: random.Random, tries: int = 5) -> Solution:
    best = None
    best_obj = None
    for _ in range(tries):
        cand = _greedy_initial(inst, demand, rng)
        if cand.unassigned:
            _repair_greedy(cand, lam, rng)
        obj = cand.objective(lam)
        if best is None or obj < best_obj:
            best, best_obj = cand, obj
    return best


def alns_solve(inst: Instance, demand: dict[int, float], lam: dict[int, float],
                iterations: int = 300, seed: int = 0, initial: Solution | None = None,
                diversify: bool = True, destroy_ops: list[str] | None = None) -> Solution:
    """diversify=False drops the occasional large destroy and multi-start
    initial construction, giving a monotone single-trajectory search --
    used for PPBRC-core, vs. full PPBRC's "bounded post-stagnation
    destroy-repair diversification" (tex, subsec:e2_algorithm)."""
    rng = random.Random(seed)
    if initial is not None:
        current = initial.clone()
    elif diversify:
        current = _multi_start_initial(inst, demand, lam, rng)
    else:
        current = _greedy_initial(inst, demand, rng)
    if current.unassigned:
        _repair_greedy(current, lam, rng)
    best = current.clone()
    best_obj = best.objective(lam)
    cur_obj = best_obj
    T0 = max(1.0, best_obj * 0.15)
    n = len(demand)
    for it in range(iterations):
        T = T0 * (0.995 ** it)
        trial = current.clone()
        # Occasional large destroy escapes structural local optima (e.g. a
        # customer-to-depot balance that small-k moves can't break out of);
        # most iterations stay small so the search doesn't thrash.
        if diversify and rng.random() < 0.15:
            k = max(2, n // 2)
        else:
            k = max(1, min(n // 4, int(rng.uniform(1, 5))))
        op = rng.choice(destroy_ops or DESTROY_OPS)
        if op == "random":
            _remove_random(trial, k, rng)
        elif op == "worst":
            _remove_worst(trial, k, lam, rng)
        elif op == "related":
            _remove_related(trial, k, rng)
        else:
            _remove_boundary(trial, k, rng)
        if rng.choice(REPAIR_OPS) == "greedy":
            _repair_greedy(trial, lam, rng)
        else:
            _repair_regret2(trial, lam, rng)
        trial_obj = trial.objective(lam)
        if trial_obj < cur_obj or rng.random() < math.exp(-(trial_obj - cur_obj) / max(T, 1e-9)):
            current = trial
            cur_obj = trial_obj
            if trial_obj < best_obj:
                best = trial.clone()
                best_obj = trial_obj
    return best


def refit_prices(inst: Instance, sol: Solution, B: dict[int, float]) -> dict:
    """Layer 2: solve the recourse at g = E(S) - B to get the next price
    vector lambda^omega.

    The primal is solved as well as the dual, because the net external
    position r_d = b_d - s_d is what the exactness certificate of
    Proposition 8 tests against: together with lambda it determines whether a
    given emission change leaves the deficit on the same piece of Psi.
    """
    E = sol.emissions()
    D = list(sol.routes.keys())
    g = {d: E[d] - B[d] for d in D}
    p_buy, p_sell = inst.prices["buy"], inst.prices["sell"]
    dual = recourse.solve_recourse_dual(g, D, inst.transfer_arcs, p_buy, p_sell)
    primal = recourse.solve_recourse_primal(g, D, inst.transfer_arcs, p_buy, p_sell)
    return {
        "lambda": dual["lambda"],
        "mu": dual["mu"],
        "g": g,
        "E": E,
        "r": recourse.net_external_position(primal, D),
    }


def certified_exact(inst: Instance, lam: dict[int, float], r: dict[int, float],
                     E_before: dict[int, float], E_after: dict[int, float]) -> bool:
    """Whether Proposition 8 certifies that the price-linearised change in
    cost between two routing plans equals the exact change. Costs O(|D|)."""
    delta = {d: E_after.get(d, 0.0) - E_before.get(d, 0.0) for d in E_after}
    return recourse.certify_same_region(lam, r, delta, inst.prices["buy"],
                                         inst.prices["sell"])


def solve_scenario(inst: Instance, demand: dict[int, float], B: dict[int, float],
                    price_rounds: int = 4, alns_iterations: int = 300, seed: int = 0,
                    initial: Solution | None = None, price_guided: bool = True,
                    diversify: bool = True, destroy_ops: list[str] | None = None) -> dict:
    """Layers 1+2 fixed-point for one scenario: alternate ALNS routing under
    the current price with a recourse-dual price refit, until stable or the
    round cap is hit. price_guided=False fixes lambda=0 for all rounds
    (single ALNS pass, no price refit) -- used by Cost-only coordination and
    SAA-ALNS, which route without depot price signals (tex, subsec:e2_algorithm)."""
    D = list(B.keys())
    if not price_guided:
        lam = {d: 0.0 for d in D}
        sol = alns_solve(inst, demand, lam, iterations=alns_iterations, seed=seed, initial=initial,
                          diversify=diversify, destroy_ops=destroy_ops)
        return {"solution": sol, "lambda": lam, "emissions": sol.emissions(),
                "distance": sum(sol.depot_metrics(d)[0] for d in D)}

    lam = {d: (inst.prices["buy"] + inst.prices["sell"]) / 2 for d in D}
    sol = initial
    last_lam = None
    prev_state = None          # (lambda, r, E) from the previous refit
    certified = tested = 0
    for r in range(price_rounds):
        sol = alns_solve(inst, demand, lam, iterations=alns_iterations, seed=seed + 1000 * r, initial=sol,
                          diversify=diversify, destroy_ops=destroy_ops)
        refit = refit_prices(inst, sol, B)

        if prev_state is not None:
            prev_lam, prev_r, prev_E = prev_state
            tested += 1
            if certified_exact(inst, prev_lam, prev_r, prev_E, refit["E"]):
                certified += 1
        prev_state = (refit["lambda"], refit["r"], refit["E"])

        new_lam = refit["lambda"]
        if last_lam is not None and all(abs(new_lam[d] - last_lam[d]) < 1e-4 for d in D):
            lam = new_lam
            break
        lam = new_lam
        last_lam = new_lam
    return {"solution": sol, "lambda": lam, "emissions": sol.emissions(),
            "distance": sum(sol.depot_metrics(d)[0] for d in D),
            "certified": certified, "certification_tests": tested}


def run_ppbrc(inst: Instance, scenarios: list[dict], outer_rounds: int = 5,
              price_rounds: int = 3, alns_iterations: int = 200, seed: int = 0,
              price_guided: bool = True, adapt_budget: bool = True, diversify: bool = True,
              destroy_ops: list[str] | None = None) -> dict:
    """price_guided=False + adapt_budget=True (with only one outer round
    needed, since routing never responds to B) is Cost-only coordination.
    price_guided=True + adapt_budget=False is Price-guided-routing-only.
    diversify=False is PPBRC-core (monotone single trajectory); the default
    (all True) is full PPBRC. destroy_ops restricts the ALNS operator set
    (SAA-ALNS uses random/worst/related only, per its tex description --
    no boundary-customer operator, which is PPBRC's own addition)."""
    D = [d["idx"] for d in inst.depots]
    pi = {w: scenarios[w]["probability"] for w in range(len(scenarios))}
    demands = {w: inst.scenario_demand(scenarios[w]) for w in range(len(scenarios))}

    B = dict(inst.depot_budget_allocation)  # tex appendix: initial B from expected reference emissions
    solutions: dict[int, Solution] = {w: None for w in demands}
    certified_total = certification_tests = 0
    effective_outer_rounds = outer_rounds if (adapt_budget and price_guided) else 1

    for outer in range(effective_outer_rounds):
        E_by_scenario = {}
        total_distance = 0.0
        for w, demand in demands.items():
            result = solve_scenario(inst, demand, B, price_rounds=price_rounds,
                                     alns_iterations=alns_iterations, seed=seed + 7919 * w + outer,
                                     initial=solutions[w], price_guided=price_guided, diversify=diversify,
                                     destroy_ops=destroy_ops)
            solutions[w] = result["solution"]
            E_by_scenario[w] = result["emissions"]
            total_distance += pi[w] * result["distance"]
            certified_total += result.get("certified", 0)
            certification_tests += result.get("certification_tests", 0)

        if not adapt_budget:
            break

        # Layer 3: exact budget master re-optimises B given fixed emissions.
        master = budget_master.solve_budget_master(
            E_by_scenario, pi, D, inst.transfer_arcs, inst.prices,
            inst.depot_budget_bounds, inst.corporate_budget, per_scenario_B=False)
        new_B = master["B"]
        moved = sum(abs(new_B[d] - B[d]) for d in D)
        B = new_B
        if moved < 1e-4 and outer > 0:
            break

    recourse_cost = sum(pi[w] * recourse.solve_recourse_primal(
        {d: E_by_scenario[w][d] - B[d] for d in D}, D, inst.transfer_arcs,
        inst.prices["buy"], inst.prices["sell"])["value"] for w in demands)
    total_objective = total_distance + recourse_cost

    return {
        "objective": total_objective,
        "distance": total_distance,
        "recourse_cost": recourse_cost,
        "B": B,
        "solutions": solutions,
        "emissions": E_by_scenario,
        "outer_rounds_used": outer + 1,
        "certified_fraction": (certified_total / certification_tests
                                if certification_tests else None),
        "certification_tests": certification_tests,
    }


SAA_ALNS_DESTROY_OPS = ["random", "worst", "related"]


def run_saa_alns(inst: Instance, scenarios: list[dict], outer_rounds: int = 5,
                  alns_iterations: int = 200, seed: int = 0) -> dict:
    """tex, subsec:e2_algorithm item 1: "a reactive destroy-repair search
    with random, worst, and related removals, greedy and regret-2 repair,
    simulated-annealing acceptance, and periodic exact optimisation of the
    shared budget" -- i.e. Cost-only coordination's mechanism (no price
    guidance) restricted to the baseline's own (narrower) operator set."""
    return run_ppbrc(inst, scenarios, outer_rounds=outer_rounds, price_rounds=1,
                      alns_iterations=alns_iterations, seed=seed, price_guided=False,
                      adapt_budget=True, diversify=True, destroy_ops=SAA_ALNS_DESTROY_OPS)

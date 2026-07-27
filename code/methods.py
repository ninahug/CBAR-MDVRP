"""Method registry for E2 (and later E3-E5): the six comparators the tex
specifies in subsec:e2_algorithm, as thin calls into ppbrc.py / ph_alns.py.
Kept in one place so every experiment harness (E2, E3, E4, E5b) dispatches
identically instead of re-deriving each method's parameter combination.
"""
from __future__ import annotations

import ph_alns
import ppbrc
from instance_io import Instance


def run_method(name: str, inst: Instance, scenarios: list[dict], seed: int,
                outer_rounds: int = 3, price_rounds: int = 2, alns_iterations: int = 150,
                ph_iterations: int = 6) -> dict:
    if name == "PPBRC":
        return ppbrc.run_ppbrc(inst, scenarios, outer_rounds=outer_rounds, price_rounds=price_rounds,
                                alns_iterations=alns_iterations, seed=seed,
                                price_guided=True, adapt_budget=True, diversify=True,
                                recombine_pool=True, lazy_rescan=True)
    if name == "PPBRC-no-recombination":
        # Full PPBRC with the final recombination switched off, so that the
        # paired difference against PPBRC measures that step alone.
        return ppbrc.run_ppbrc(inst, scenarios, outer_rounds=outer_rounds, price_rounds=price_rounds,
                                alns_iterations=alns_iterations, seed=seed,
                                price_guided=True, adapt_budget=True, diversify=True,
                                recombine_pool=False)
    if name == "PPBRC-core":
        return ppbrc.run_ppbrc(inst, scenarios, outer_rounds=outer_rounds, price_rounds=price_rounds,
                                alns_iterations=alns_iterations, seed=seed,
                                price_guided=True, adapt_budget=True, diversify=False)
    if name == "Cost-only":
        return ppbrc.run_ppbrc(inst, scenarios, outer_rounds=outer_rounds, price_rounds=price_rounds,
                                alns_iterations=alns_iterations, seed=seed,
                                price_guided=False, adapt_budget=True, diversify=True)
    if name == "Price-guided-only":
        return ppbrc.run_ppbrc(inst, scenarios, outer_rounds=outer_rounds, price_rounds=price_rounds,
                                alns_iterations=alns_iterations, seed=seed,
                                price_guided=True, adapt_budget=False, diversify=True)
    if name == "SAA-ALNS":
        return ppbrc.run_saa_alns(inst, scenarios, outer_rounds=outer_rounds,
                                   alns_iterations=alns_iterations, seed=seed)
    if name == "PH-ALNS":
        return ph_alns.run_ph_alns(inst, scenarios, ph_iterations=ph_iterations, price_rounds=price_rounds,
                                    alns_iterations=alns_iterations, seed=seed)
    raise ValueError(f"unknown method {name!r}")


ALL_METHODS = ["SAA-ALNS", "PH-ALNS", "Cost-only", "Price-guided-only",
               "PPBRC-core", "PPBRC-no-recombination", "PPBRC"]
FROZEN_SEEDS = [91001, 91002, 91003, 91004, 91005]


def mechanism_diagnostics(inst: Instance, result: dict) -> dict:
    """Table mechanism_activation columns, computed from a method result's
    final (emissions, B) by re-solving the recourse primal/dual per
    scenario: deficit-surplus coexistence, mean price dispersion, transfer
    use, saturated-arc incidence, and external purchase/sale incidence."""
    import recourse
    D = [d["idx"] for d in inst.depots]
    B = result["B"]
    E = result["emissions"]
    n_scenarios = len(E)
    n_arcs = len(inst.transfer_arcs)

    coexist = 0
    transfer_used = 0
    purchase = 0
    sale = 0
    saturated = 0
    dispersion_sum = 0.0
    for w, Ew in E.items():
        g = {d: Ew[d] - B[d] for d in D}
        deficit = any(v > 1e-6 for v in g.values())
        surplus = any(v < -1e-6 for v in g.values())
        if deficit and surplus:
            coexist += 1
        rec = recourse.solve_recourse(g, D, inst.transfer_arcs, inst.prices)
        if any(v > 1e-6 for v in rec["primal"]["t"].values()):
            transfer_used += 1
        if any(v > 1e-6 for v in rec["primal"]["b"].values()):
            purchase += 1
        if any(v > 1e-6 for v in rec["primal"]["s"].values()):
            sale += 1
        for arc in inst.transfer_arcs:
            key = (arc["from"], arc["to"])
            if rec["primal"]["t"][key] > arc["capacity"] - 1e-6:
                saturated += 1
        lam = rec["dual"]["lambda"]
        dispersion_sum += (max(lam.values()) - min(lam.values())) if lam else 0.0

    return {
        "deficit_surplus_coexistence_pct": 100.0 * coexist / n_scenarios,
        "mean_price_dispersion": dispersion_sum / n_scenarios,
        "transfer_use_pct": 100.0 * transfer_used / n_scenarios,
        "saturated_arcs_pct": 100.0 * saturated / max(1, n_scenarios * n_arcs),
        "purchase_scenarios_pct": 100.0 * purchase / n_scenarios,
        "sale_scenarios_pct": 100.0 * sale / n_scenarios,
    }

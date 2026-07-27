"""Adapts the raw Cordeau JSON schema (written by
gpt_test/build_cordeau_cbar_benchmark.py, asdict(CordeauInstance)) into the
same duck-typed interface instance_io.Instance exposes, so ppbrc.py's
route_metrics/alns_solve/Solution machinery can be reused unchanged for the
E5a classic-MDVRP regression instead of writing a second routing engine.

E5a has no carbon layer and no restricted eligibility: every customer is
served by whichever depot the solver assigns it to, so eligible_depots is
every depot index (matching the classic MDVRP's free multi-depot
assignment, unlike the CBAR suites' Omega_i restriction).
"""
from __future__ import annotations

import json
from pathlib import Path


class PseudoDepot(dict):
    pass


class PseudoInstance:
    def __init__(self, problem: dict, name: str):
        self.name = name
        self.suite = "E5a"
        p = problem
        n_depots = p["n_depots"]
        # Cordeau raw-format convention: max_duration == 0 means "no route
        # duration limit", not "zero distance allowed" -- literally passing
        # 0 through made every nonempty route infeasible (route_metrics'
        # feasibility check is distance <= max_duration), which silently
        # produced a fully empty, 0-distance "solution" until caught here.
        def duration(d):
            v = p["max_duration"][d]
            return v if v > 0 else float("inf")

        self.depots = [
            {"idx": d, "x": p["depots"][d]["x"], "y": p["depots"][d]["y"],
             "capacity": p["capacity"][d], "n_vehicles": p["max_vehicles"],
             "max_duration": duration(d)}
            for d in range(n_depots)
        ]
        self.customers = [
            {"idx": c["node_id"] - 1, "x": c["x"], "y": c["y"], "service": c.get("service", 0.0),
             "base_demand": c["demand"], "home_depot": 0, "eligible_depots": list(range(n_depots))}
            for c in p["customers"]
        ]
        self.n_depots = n_depots
        self.n_customers = len(self.customers)
        self.eligible = {c["idx"]: c["eligible_depots"] for c in self.customers}
        self.transfer_arcs: list[dict] = []
        self.prices = {"buy": 0.0, "sell": 0.0}
        self.theta = 1.0

        class V:
            __slots__ = ("idx", "depot", "capacity", "max_duration")

        self.vehicles = []
        for d in range(n_depots):
            for _ in range(p["max_vehicles"]):
                v = V()
                v.idx = len(self.vehicles)
                v.depot = d
                v.capacity = p["capacity"][d]
                v.max_duration = p["max_duration"][d]
                self.vehicles.append(v)
        self.customers_of_vehicle = {v.idx: [c["idx"] for c in self.customers] for v in self.vehicles}

    def depot_xy(self, d: int):
        dep = self.depots[d]
        return dep["x"], dep["y"]

    def customer_xy(self, i: int):
        c = self.customers[i]
        return c["x"], c["y"]

    def scenario_demand(self, scenario) -> dict[int, float]:
        return {c["idx"]: c["base_demand"] for c in self.customers}


def load_e5a(path: str | Path) -> PseudoInstance:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return PseudoInstance(raw["problem"], raw["source_instance"])


class PseudoCbarInstance(PseudoInstance):
    """E5b: PseudoInstance plus the carbon overlay (base.json + a
    (budget_factor, capacity_scale) variant file). Customer eligibility is
    unrestricted across all public depots, per the tex's E5b paragraph
    ("Customer eligibility remains unrestricted across the public depots, as
    in the classical MDVRP") -- already PseudoInstance's default.
    """

    def __init__(self, base: dict, overlay: dict):
        super().__init__(base["retained_public_data"], base["source_instance"])
        self.suite = "E5b"
        # gpt_test/build_cordeau_cbar_benchmark.py indexes depots 1-based
        # (Cordeau's own convention, inherited from parse_solution_routes /
        # per_depot_reference_demand); PseudoInstance.depots is 0-based
        # ("idx" starts at 0). Remap every depot-keyed field from the
        # overlay/base to 0-based here so run_ppbrc's D = [d["idx"] ...]
        # actually matches these dicts' keys instead of KeyError-ing.
        self.transfer_arcs = [
            {**a, "from": a["from"] - 1, "to": a["to"] - 1} for a in overlay["transfer_arcs"]
        ]
        self.prices = base["prices"]
        self.theta = base["emission_parameters"]["theta"]
        # Without this the E5b runs would price fuel at zero while every
        # synthetic suite prices it at 0.1 (ppbrc reads it via getattr with a
        # 0.0 default, so the omission was silent).
        self.emission_cost = base.get("emission_cost", 0.0)
        self.reference_demand = {
            int(k) - 1: v for k, v in base.get("reference_demand_per_depot", {}).items()
        }
        self.corporate_budget = overlay["corporate_budget"]
        self.depot_budget_allocation = {int(k) - 1: v for k, v in overlay["depot_budget_allocation"].items()}
        self.depot_budget_bounds = {int(k) - 1: tuple(v) for k, v in overlay["depot_budget_bounds"].items()}
        self._scenarios = base["scenarios"]

    def scenarios(self, bank: str = "train", prefix: int | None = None) -> list[dict]:
        return self._scenarios.get(bank, [])

    def scenario_demand(self, scenario: dict) -> dict[int, float]:
        # Scenario demand keys are Cordeau 1-based node_id strings (written
        # by correlated_scenarios() using c.node_id); convert to the 0-based
        # customer idx convention used everywhere else in this codebase.
        return {int(k) - 1: v for k, v in scenario["demand"].items()}


def load_e5b(instance_dir: str | Path, variant: str) -> PseudoCbarInstance:
    d = Path(instance_dir)
    base = json.loads((d / "base.json").read_text(encoding="utf-8"))
    overlay = json.loads((d / f"{variant}.json").read_text(encoding="utf-8"))
    return PseudoCbarInstance(base, overlay)


def parse_bks_distance(solution_path: str | Path) -> float | None:
    """First token of a Cordeau .res file's first line is the total route
    cost (see gpt_test/build_cordeau_cbar_benchmark.py:parse_solution_routes,
    reimplemented minimally here to avoid importing a script module)."""
    p = Path(solution_path)
    if not p.exists():
        return None
    first_line = p.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    return float(first_line.split()[0])

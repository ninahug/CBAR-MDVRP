"""Loads a generated instance JSON (from build_data_release.py or the E5
Cordeau adapter) into plain Python structures convenient for exact_model.py,
recourse.py, and the PPBRC heuristic.

Fields not produced by the data generator but needed by the deterministic-
equivalent MILP (tex eq. obj) are filled with documented defaults:
  - dispatch cost F_k = 0 (no per-vehicle fixed cost is generated)
  - travel cost c_ij = Euclidean distance d_ij (unit cost per distance)
  - travel time tau_ij = d_ij (unit speed)
  - service time sigma_i = 0 (already zero in the generator's customers)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Vehicle:
    idx: int          # global vehicle index
    depot: int
    capacity: float
    max_duration: float
    dispatch_cost: float = 0.0


def euclid(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


class Instance:
    def __init__(self, raw: dict):
        self.raw = raw
        self.name = raw.get("name", raw.get("source_instance", "?"))
        self.suite = raw.get("suite", raw.get("experiment", "?"))
        self.depots = raw["depots"]              # list of dicts: idx,x,y,capacity,n_vehicles,max_duration
        self.customers = raw["customers"]        # list of dicts: idx,x,y,service,base_demand,home_depot,eligible_depots
        self.transfer_arcs = raw.get("transfer_arcs", [])
        self.reference_demand = {int(k): v for k, v in raw.get("reference_demand", {}).items()}
        self.emission_cost = raw.get("emission_cost", 0.0)
        self.corporate_budget = raw.get("corporate_budget")
        self.depot_budget_allocation = {int(k): v for k, v in raw.get("depot_budget_allocation", {}).items()}
        self.depot_budget_bounds = {int(k): tuple(v) for k, v in raw.get("depot_budget_bounds", {}).items()}
        self.prices = raw.get("prices", {"buy": 9.0, "sell": 2.0})
        self.theta = 1.0
        self.rho_empty = 0.9
        self.rho_full = 1.5

        self.n_depots = len(self.depots)
        self.n_customers = len(self.customers)

        self.eligible = {c["idx"]: list(c.get("eligible_depots", [c["home_depot"]])) for c in self.customers}

        self.vehicles: list[Vehicle] = []
        for d in self.depots:
            for _ in range(int(d["n_vehicles"])):
                self.vehicles.append(Vehicle(len(self.vehicles), d["idx"], d["capacity"], d["max_duration"]))

        self.customers_of_vehicle = {
            v.idx: [c["idx"] for c in self.customers if v.depot in self.eligible[c["idx"]]]
            for v in self.vehicles
        }

    def depot_xy(self, d: int):
        dep = self.depots[d]
        return dep["x"], dep["y"]

    def customer_xy(self, i: int):
        c = self.customers[i]
        return c["x"], c["y"]

    def node_xy(self, node) -> tuple[float, float]:
        # node is ("depot", d) or a customer index (int)
        if isinstance(node, tuple):
            return self.depot_xy(node[1])
        return self.customer_xy(node)

    def distance(self, a, b) -> float:
        ax, ay = self.node_xy(a)
        bx, by = self.node_xy(b)
        return euclid(ax, ay, bx, by)

    def scenarios(self, bank: str = "train", prefix: int | None = None) -> list[dict]:
        bank_data = self.raw["scenarios"].get(bank)
        if bank_data is None:
            return []
        if isinstance(bank_data, dict):  # E3 nested prefixes: {"20": [...], "50": [...], "100": [...]}
            key = str(prefix) if prefix is not None else max(bank_data, key=lambda k: int(k))
            return bank_data[key]
        return bank_data

    def scenario_demand(self, scenario: dict) -> dict[int, float]:
        return {int(k): v for k, v in scenario["demand"].items()}


def load_instance(path: str | Path) -> Instance:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Instance(raw)

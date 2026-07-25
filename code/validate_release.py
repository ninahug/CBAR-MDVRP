#!/usr/bin/env python3
"""Automated validation for the CBAR-MDVRP-DATA-v1.0 synthetic release.

Implements the checks named in the "Release structure and validation"
paragraph: dimensions, positivity, individual-vehicle feasibility, fleet
slack, budget-bound feasibility, stream separation, and checksums. "An
instance failing any validation is regenerated before algorithms are run;
it is not silently removed after observing algorithm performance" -- this
script only reports failures, it does not delete or edit instances.

Usage:
    python validate_release.py --out CBAR_MDVRP_DATA_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_instance(inst: dict) -> list[str]:
    errors = []
    name = f"{inst.get('suite')}/{inst.get('name')}"

    depots = inst["depots"]
    customers = inst["customers"]

    # Dimensions.
    if len(depots) != inst["n_depots"]:
        errors.append(f"{name}: depot count {len(depots)} != n_depots {inst['n_depots']}")
    if len(customers) != inst["n_customers"]:
        errors.append(f"{name}: customer count {len(customers)} != n_customers {inst['n_customers']}")

    # Positivity: coordinates in-region, positive base demand and capacity.
    for c in customers:
        if not (0.0 <= c["x"] <= 10.0 and 0.0 <= c["y"] <= 10.0):
            errors.append(f"{name}: customer {c['idx']} coordinate out of region")
        if c["base_demand"] <= 0:
            errors.append(f"{name}: customer {c['idx']} base_demand not positive")
    for d in depots:
        if d["capacity"] <= 0 or d["n_vehicles"] <= 0:
            errors.append(f"{name}: depot {d['idx']} has non-positive capacity/fleet")

    vehicle_capacity = min(d["capacity"] for d in depots) if depots else 0.0

    # Individual-vehicle feasibility + positivity of every scenario demand.
    def check_bank(bank_name: str, scenarios: list[dict]):
        for sc in scenarios:
            for cust_id, q in sc["demand"].items():
                if q <= 0:
                    errors.append(f"{name}: {bank_name} scenario {sc['scenario']} customer {cust_id} demand <= 0")
                if q > vehicle_capacity + 1e-6:
                    errors.append(
                        f"{name}: {bank_name} scenario {sc['scenario']} customer {cust_id} "
                        f"demand {q:.3f} exceeds vehicle capacity {vehicle_capacity}")

    scenarios = inst["scenarios"]
    for bank_name, bank in scenarios.items():
        if bank_name == "train" and isinstance(bank, dict):
            for prefix_name, prefix_scenarios in bank.items():
                check_bank(f"train[{prefix_name}]", prefix_scenarios)
        else:
            check_bank(bank_name, bank)

    # Fleet slack under mean/base demand (no scenario shock).
    by_depot: dict[str, float] = {}
    for c in customers:
        by_depot[str(c["home_depot"])] = by_depot.get(str(c["home_depot"]), 0.0) + c["base_demand"]
    for d in depots:
        total = by_depot.get(str(d["idx"]), 0.0)
        capacity_total = d["capacity"] * d["n_vehicles"]
        if total > capacity_total + 1e-6:
            errors.append(f"{name}: depot {d['idx']} base demand {total:.2f} exceeds fleet capacity {capacity_total:.2f}")

    # Budget-bound feasibility: sum of lower bounds <= B_corp <= sum of upper bounds.
    bounds = inst["depot_budget_bounds"]
    lower_sum = sum(v[0] for v in bounds.values())
    upper_sum = sum(v[1] for v in bounds.values())
    B_corp = inst["corporate_budget"]
    if not (lower_sum - 1e-6 <= B_corp <= upper_sum + 1e-6):
        errors.append(f"{name}: B_corp {B_corp:.3f} outside [{lower_sum:.3f}, {upper_sum:.3f}]")

    # Transfer-arc capacity must be non-negative and friction non-negative,
    # and no directed arc should appear more than once (e.g. a degenerate
    # adjacency generator listing the same undirected edge twice).
    seen_arcs = set()
    for arc in inst["transfer_arcs"]:
        if arc["capacity"] < 0 or arc["friction"] < 0:
            errors.append(f"{name}: transfer arc {arc['from']}->{arc['to']} has negative capacity/friction")
        key = (arc["from"], arc["to"])
        if key in seen_arcs:
            errors.append(f"{name}: duplicate transfer arc {arc['from']}->{arc['to']}")
        seen_arcs.add(key)

    return errors


def check_stream_separation(instances: list[dict]) -> list[str]:
    """Two instances may legitimately share a seed set (e.g. E2's three
    carbon-network strata are, by design, the same customers/depots/demand
    under three different budget/capacity overlays -- see
    suite_config.build_e2_specs). What must never happen is two instances
    sharing a seed while disagreeing on the geometry/demand it produced,
    which would mean an accidental seed collision rather than an
    intentional shared group."""
    errors = []
    by_seed_tuple: dict[tuple, list[dict]] = {}
    for inst in instances:
        key = tuple(sorted(inst["seeds"].items()))
        by_seed_tuple.setdefault(key, []).append(inst)

    for key, group in by_seed_tuple.items():
        if len(group) < 2:
            continue
        ref = group[0]
        ref_geo = (ref["depots"], ref["customers"])
        for other in group[1:]:
            if (other["depots"], other["customers"]) != ref_geo:
                errors.append(
                    f"seed collision: {ref['suite']}/{ref['name']} and "
                    f"{other['suite']}/{other['name']} share seeds {dict(key)} "
                    f"but generated different geometry/demand")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("CBAR_MDVRP_DATA_v1"))
    args = ap.parse_args()

    manifest_path = args.out / "release_manifest.csv"
    with manifest_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    all_errors: list[str] = []
    instances = []
    for row in rows:
        path = args.out / row["file"]
        data = path.read_bytes()
        digest = sha256_bytes(data)
        if digest != row["sha256"]:
            all_errors.append(f"{row['suite']}/{row['name']}: checksum mismatch (manifest says {row['sha256']}, file is {digest})")
        inst = json.loads(data)
        instances.append(inst)
        all_errors.extend(check_instance(inst))

    all_errors.extend(check_stream_separation(instances))

    print(f"checked {len(instances)} instances")
    if all_errors:
        print(f"\n{len(all_errors)} FAILURES:")
        for e in all_errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print("all checks PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Builds the frozen CBAR-MDVRP-DATA-v1.0 synthetic release: the 100
tuning/E1/E2/E3/E4 instances described in Table "frozen_data_suites" and
Appendix "Instance and scenario generation" of
t-mdvrp_CBAR_PPBRC_public_benchmark_protocol.tex.

The E5 public-benchmark overlay (Cordeau instances) is a separate pipeline;
see gpt_test/build_cordeau_cbar_benchmark.py. This script does not touch it.

Usage:
    python build_data_release.py --out CBAR_MDVRP_DATA_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import carbon
import demand
import geometry
import network
from suite_config import SuiteInstanceSpec, all_specs

# tex, "Demand streams": exact stream bases, "the same instance index is
# added to each base, so every stream can be reproduced from the public seed
# manifest."
GEOMETRY_BASE = 1_000_000
TRAIN_BASE = 2_000_000       # tex calls this the "optimisation" stream
VALID_BASE = 3_000_000
TEST_BASE = 4_000_000
SHIFT_BASE = 5_000_000

PRICES = {"buy": 9.0, "sell": 2.0}  # tex, subsec:experimental_setup


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_instance(spec: SuiteInstanceSpec) -> dict:
    geo_rng = np.random.default_rng(GEOMETRY_BASE + spec.seed_index)
    depots = geometry.place_depots(spec.n_depots, geo_rng)
    customers = geometry.generate_customers(depots, spec.n_customers, spec.boundary_share, geo_rng)
    demand.base_demands(customers, depots, geometry.BASE_DEMAND_CV,
                         geometry.DEPOT_SCALE_CV, geo_rng, geometry.BASE_MEAN_DEMAND)
    geometry.generate_fleet(depots, customers)

    n_depots = spec.n_depots

    def gen_bank(n_scen: int, base: int, **kwargs) -> list[dict]:
        return demand.scenario_generator(
            customers, n_depots, n_scen, base + spec.seed_index,
            spec.demand_cv_common, spec.demand_cv_idio, spec.cross_depot_corr,
            geometry.VEHICLE_CAPACITY, **kwargs)

    scenarios: dict = {}
    if spec.nested_prefixes:
        bank = gen_bank(max(spec.nested_prefixes), TRAIN_BASE)
        # Nested by construction: the generator draws scenarios sequentially
        # from one rng stream, so a size-20 prefix of the size-100 bank is
        # identical to an independent size-20 draw from the same seed.
        scenarios["train"] = {str(n): bank[:n] for n in spec.nested_prefixes}
    else:
        scenarios["train"] = gen_bank(spec.train_scenarios, TRAIN_BASE)

    for bank_name, base in (("validation", VALID_BASE), ("test", TEST_BASE)):
        n = spec.eval_banks.get(bank_name)
        if n:
            scenarios[bank_name] = gen_bank(n, base)

    if spec.eval_banks.get("shift"):
        scenarios["shift"] = gen_bank(
            spec.eval_banks["shift"], SHIFT_BASE, student_t=True,
            regime_prob=0.05, regime_multiplier=1.5)

    E0 = carbon.reference_emission_per_depot(depots, customers)
    B_corp, alloc, bounds = carbon.corporate_budget(E0, spec.budget_factor)
    adjacency = geometry.ring_adjacency(n_depots)
    arcs = network.build_transfer_network(depots, adjacency, spec.capacity_scale, E0)

    return {
        "benchmark_version": "CBAR-MDVRP-DATA-v1.0",
        "suite": spec.suite,
        "name": spec.name,
        "seed_index": spec.seed_index,
        "n_customers": spec.n_customers,
        "n_depots": spec.n_depots,
        "seeds": {
            "geometry": GEOMETRY_BASE + spec.seed_index,
            "train": TRAIN_BASE + spec.seed_index,
            "validation": VALID_BASE + spec.seed_index,
            "test": TEST_BASE + spec.seed_index,
            "shift": SHIFT_BASE + spec.seed_index,
        },
        "factors": {
            "budget_factor": spec.budget_factor,
            "capacity_scale": spec.capacity_scale,
            "demand_cv_common": spec.demand_cv_common,
            "demand_cv_idio": spec.demand_cv_idio,
            "cross_depot_corr": spec.cross_depot_corr,
            "boundary_share": spec.boundary_share,
        },
        "depots": [vars(d) for d in depots],
        "customers": [
            {**vars(c), "eligible_depots": list(c.eligible_depots)} for c in customers
        ],
        "transfer_arcs": arcs,
        "reference_emission": {str(k): v for k, v in E0.items()},
        "corporate_budget": B_corp,
        "depot_budget_allocation": {str(k): v for k, v in alloc.items()},
        "depot_budget_bounds": {str(k): list(v) for k, v in bounds.items()},
        "prices": PRICES,
        "scenarios": scenarios,
    }


def write_json(path: Path, obj: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(r"D:\CBAR_MDVRP\data\generated\synthetic_v1"))
    ap.add_argument("--suites", nargs="*", default=None,
                     help="restrict to a subset of suites, e.g. --suites E1 E2")
    args = ap.parse_args()

    specs = all_specs()
    if args.suites:
        wanted = set(args.suites)
        specs = [s for s in specs if s.suite in wanted]

    manifest_rows = []
    for spec in specs:
        inst = build_instance(spec)
        path = args.out / spec.suite / f"{spec.name}.json"
        digest = write_json(path, inst)
        manifest_rows.append({
            "suite": spec.suite,
            "name": spec.name,
            "seed_index": spec.seed_index,
            "n_customers": spec.n_customers,
            "n_depots": spec.n_depots,
            "file": str(path.relative_to(args.out)),
            "sha256": digest,
        })
        print(f"built {spec.suite}/{spec.name}  "
              f"(C={spec.n_customers}, D={spec.n_depots}, seed={spec.seed_index})")

    manifest_path = args.out / "release_manifest.csv"
    args.out.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        w.writeheader()
        w.writerows(manifest_rows)

    write_json(args.out / "release_summary.json", {
        "benchmark_version": "CBAR-MDVRP-DATA-v1.0",
        "total_instances": len(manifest_rows),
        "by_suite": {
            suite: sum(1 for r in manifest_rows if r["suite"] == suite)
            for suite in sorted({r["suite"] for r in manifest_rows})
        },
    })
    print(f"\nwrote {len(manifest_rows)} instances to {args.out}")


if __name__ == "__main__":
    main()

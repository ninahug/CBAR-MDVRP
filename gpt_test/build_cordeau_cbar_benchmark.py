#!/usr/bin/env python3
"""Build the frozen public Cordeau-MDVRP extension for CBAR-MDVRP.

The script accepts the original Cordeau instance and solution archives or
extracted folders. It creates:
  * E5a unmodified deterministic MDVRP JSON files;
  * E5b public-backbone stochastic CBAR JSON files;
  * source/conversion manifests with SHA-256 hashes.

It intentionally does not silently download data unless --download is passed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import tempfile
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

INSTANCE_URL = "https://neo.lcc.uma.es/vrp/wp-content/data/instances/cordeau/C-mdvrp.zip"
SOLUTION_URL = "https://neo.lcc.uma.es/vrp/wp-content/data/instances/cordeau/C-mdvrp-sol.zip"
SELECTED_E5B = ["p01", "p02", "p03", "p12", "p04", "p05", "p06", "p07", "p15", "p18", "p21"]
# Service-cover factors against expected total demand; 1.00 stocks the
# network to its mean requirement. See suite_config.py.
BUDGET_FACTORS = [0.90, 1.00, 1.10]
CAPACITY_SCALES = [0.60, 1.00, 1.40]
P_BUY, P_SELL = 2.0, 0.5
EMISSION_COST = 0.1  # load-dependent fuel priced as operating cost
TRAIN_N, VALID_N, TEST_N = 50, 200, 800
STREAM_BASE = {"train": 12_000_000, "validation": 13_000_000, "test": 14_000_000}


@dataclass(frozen=True)
class Node:
    node_id: int
    x: float
    y: float
    service: float
    demand: float


@dataclass
class CordeauInstance:
    name: str
    problem_type: int
    max_vehicles: int
    n_customers: int
    n_depots: int
    max_duration: list[float]
    capacity: list[float]
    customers: list[Node]
    depots: list[Node]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r, target.open("wb") as f:
        shutil.copyfileobj(r, f)


def extract_zip(path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(out)


def _clean_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.replace("\r", "").split("\n") if ln.strip()]


def parse_cordeau_instance(path: Path) -> CordeauInstance:
    lines = _clean_lines(path.read_text(encoding="utf-8", errors="replace"))
    if not lines:
        raise ValueError(f"empty instance: {path}")
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"invalid Cordeau header in {path}: {lines[0]}")
    ptype, max_vehicles, n, t = map(int, first[:4])
    if ptype != 2:
        raise ValueError(f"{path.name}: expected MDVRP type 2, got {ptype}")
    durations, capacities = [], []
    cursor = 1
    for _ in range(t):
        vals = lines[cursor].split()
        durations.append(float(vals[0]))
        capacities.append(float(vals[1]))
        cursor += 1
    records: list[Node] = []
    for row in lines[cursor:cursor + n + t]:
        vals = row.split()
        if len(vals) < 5:
            raise ValueError(f"bad node row in {path}: {row}")
        records.append(Node(int(float(vals[0])), float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])))
    if len(records) != n + t:
        raise ValueError(f"{path.name}: expected {n+t} nodes, got {len(records)}")
    return CordeauInstance(
        name=path.stem.lower(), problem_type=ptype, max_vehicles=max_vehicles,
        n_customers=n, n_depots=t, max_duration=durations, capacity=capacities,
        customers=records[:n], depots=records[n:]
    )


def find_files(root: Path, allowed: set[str] | None = None) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        key = p.stem.lower() if p.suffix else p.name.lower()
        if allowed is None or key in allowed:
            found[key] = p
    return found


def parse_solution_routes(path: Path) -> tuple[float, list[tuple[int, list[int]]]]:
    lines = _clean_lines(path.read_text(encoding="utf-8", errors="replace"))
    if not lines:
        raise ValueError("empty solution")
    bks = float(lines[0].split()[0])
    routes: list[tuple[int, list[int]]] = []
    for line in lines[1:]:
        # Remove optional service-time annotations such as 12(45.3).
        clean = re.sub(r"\([^)]*\)", "", line)
        tok = clean.split()
        if len(tok) < 5:
            continue
        depot = int(float(tok[0]))
        customers = []
        for x in tok[4:]:
            try:
                customers.append(int(float(x)))
            except ValueError:
                pass
        if customers:
            routes.append((depot, customers))
    return bks, routes


def euclid(a: Node, b: Node) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def route_emission(inst: CordeauInstance, depot_index_1: int, route: Sequence[int], demands: dict[int, float],
                   rho0: float = 0.9, rhoQ: float = 1.5, theta: float = 1.0) -> float:
    depot = inst.depots[depot_index_1 - 1]
    node_map = {n.node_id: n for n in inst.customers}
    sequence = [depot] + [node_map[i] for i in route if i in node_map] + [depot]
    remaining = sum(demands.get(i, node_map[i].demand) for i in route if i in node_map)
    cap = inst.capacity[depot_index_1 - 1]
    e = 0.0
    for a, b in zip(sequence[:-1], sequence[1:]):
        factor = rho0 + (rhoQ - rho0) * max(0.0, min(cap, remaining)) / cap
        e += theta * euclid(a, b) * factor
        if b.node_id in demands:
            remaining -= demands[b.node_id]
    return e


def bks_reference_emission(inst: CordeauInstance, routes: list[tuple[int, list[int]]]) -> float:
    demands = {n.node_id: n.demand for n in inst.customers}
    return sum(route_emission(inst, d, r, demands) for d, r in routes)


def per_depot_reference_demand(inst: CordeauInstance) -> dict[int, float]:
    """Base demand homed at each depot (1-based depot index).

    The synthetic generator sets a customer's home depot explicitly and sums
    base demand over it (code/carbon.py: reference_demand_per_depot). Cordeau
    instances carry no home-depot field, so the nearest depot plays that role,
    which is the same rule the generator's core-customer sampling follows.
    This depends only on published coordinates and demands, so unlike the old
    emission reference it never depends on a solution file.
    """
    per_depot: dict[int, float] = {d: 0.0 for d in range(1, inst.n_depots + 1)}
    for c in inst.customers:
        d = min(range(1, inst.n_depots + 1), key=lambda j: euclid(c, inst.depots[j - 1]))
        per_depot[d] += c.demand
    return per_depot


def nearest_depot_reference(inst: CordeauInstance) -> tuple[list[tuple[int, list[int]]], float]:
    """Frozen deterministic fallback, explicitly not a BKS route.

    Assign each customer to its nearest depot, sort by polar angle, and split by
    capacity. This is used only when a public solution file cannot be parsed.
    """
    buckets: dict[int, list[Node]] = {d: [] for d in range(1, inst.n_depots + 1)}
    for c in inst.customers:
        d = min(range(1, inst.n_depots + 1), key=lambda j: euclid(c, inst.depots[j - 1]))
        buckets[d].append(c)
    routes: list[tuple[int, list[int]]] = []
    for d, nodes in buckets.items():
        depot = inst.depots[d - 1]
        nodes.sort(key=lambda c: math.atan2(c.y - depot.y, c.x - depot.x))
        current, load = [], 0.0
        for c in nodes:
            if current and load + c.demand > inst.capacity[d - 1] + 1e-9:
                routes.append((d, [x.node_id for x in current]))
                current, load = [], 0.0
            current.append(c); load += c.demand
        if current:
            routes.append((d, [x.node_id for x in current]))
    return routes, bks_reference_emission(inst, routes)


def depot_assignment(inst: CordeauInstance) -> dict[int, int]:
    return {c.node_id: min(range(inst.n_depots), key=lambda j: euclid(c, inst.depots[j])) for c in inst.customers}


def correlated_scenarios(inst: CordeauInstance, n_scen: int, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    d = inst.n_depots
    rho = 0.25
    corr = np.full((d, d), rho); np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)
    assign = depot_assignment(inst)
    sigma_common = math.sqrt(math.log(1.0 + 0.25 ** 2))
    sigma_idio = math.sqrt(math.log(1.0 + 0.10 ** 2))
    scenarios = []
    for s in range(n_scen):
        for _attempt in range(1000):
            z = L @ rng.standard_normal(d)
            eps = rng.standard_normal(inst.n_customers)
            q: dict[str, float] = {}
            total = 0.0
            feasible = True
            for idx, c in enumerate(inst.customers):
                mult = math.exp(sigma_common * z[assign[c.node_id]] - 0.5 * sigma_common ** 2)
                mult *= math.exp(sigma_idio * eps[idx] - 0.5 * sigma_idio ** 2)
                val = max(0.01, c.demand * mult)
                if val > max(inst.capacity) + 1e-9:
                    feasible = False; break
                q[str(c.node_id)] = val; total += val
            if feasible and total <= inst.max_vehicles * sum(inst.capacity) + 1e-9:
                scenarios.append({"scenario": s + 1, "probability": 1.0 / n_scen, "demand": q})
                break
        else:
            raise RuntimeError(f"failed to sample feasible scenario for {inst.name}")
    return scenarios


def mst_plus_nearest(inst: CordeauInstance) -> list[dict[str, float]]:
    n = inst.n_depots
    if n <= 1:
        return []
    dist = [[euclid(inst.depots[i], inst.depots[j]) for j in range(n)] for i in range(n)]
    selected = {0}; edges: set[tuple[int, int]] = set()
    while len(selected) < n:
        i, j = min(((i, j) for i in selected for j in range(n) if j not in selected), key=lambda x: dist[x[0]][x[1]])
        edges.add(tuple(sorted((i, j)))); selected.add(j)
    for i in range(n):
        candidates = sorted((dist[i][j], j) for j in range(n) if j != i and tuple(sorted((i, j))) not in edges)
        if candidates:
            edges.add(tuple(sorted((i, candidates[0][1]))))
    nonzero = [dist[i][j] for i, j in edges]
    scale = sum(nonzero) / len(nonzero)
    arcs = []
    for i, j in sorted(edges):
        kappa = 0.15 + 0.35 * dist[i][j] / scale
        arcs.extend([
            {"from": i + 1, "to": j + 1, "distance": dist[i][j], "friction": kappa},
            {"from": j + 1, "to": i + 1, "distance": dist[i][j], "friction": kappa},
        ])
    return arcs


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def build(instances_root: Path, solutions_root: Path | None, out: Path) -> None:
    instance_files = find_files(instances_root)
    if not instance_files:
        raise RuntimeError(f"no files found under {instances_root}")
    solution_files = find_files(solutions_root) if solutions_root else {}
    out.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    parsed: dict[str, CordeauInstance] = {}
    for name, path in sorted(instance_files.items()):
        try:
            inst = parse_cordeau_instance(path)
        except Exception:
            continue
        parsed[name] = inst
        e5a = {
            "benchmark_version": "CBAR-MDVRP-PUBLIC-v1.0",
            "experiment": "E5a-classic-MDVRP-regression",
            "source_instance": name,
            "carbon_enabled": False,
            "uncertainty_enabled": False,
            "problem": asdict(inst),
        }
        e5a_path = out / "E5a_classic" / f"{name}.json"
        write_json(e5a_path, e5a)
        manifest_rows.append({"experiment": "E5a", "instance": name, "variant": "classic", "file": str(e5a_path.relative_to(out)), "sha256": sha256(e5a_path), "reference_route": "not-required"})

    missing = [x for x in SELECTED_E5B if x not in parsed]
    if missing:
        raise RuntimeError(f"selected public instances missing: {missing}")
    for idx, name in enumerate(SELECTED_E5B, start=1):
        inst = parsed[name]
        ref_type = "fallback"
        routes: list[tuple[int, list[int]]]
        published_bks = None
        if name in solution_files:
            try:
                published_bks, routes = parse_solution_routes(solution_files[name])
                ref_e = bks_reference_emission(inst, routes)
                ref_type = "public-bks-route"
            except Exception:
                routes, ref_e = nearest_depot_reference(inst)
        else:
            routes, ref_e = nearest_depot_reference(inst)
        graph = mst_plus_nearest(inst)
        # The routes above are retained only as provenance for E5a and for the
        # published-distance column; the E5b reference no longer reads them.
        per_depot_r0 = per_depot_reference_demand(inst)
        ref_r = sum(per_depot_r0.values())

        # Geometry, eligibility, and scenario draws do not depend on
        # (budget_factor, capacity_scale); computing and storing them once
        # per instance instead of once per of the 9 grid cells avoids both
        # redundant correlated_scenarios() sampling (9x wasted compute) and
        # redundant on-disk storage of identical 50+200+800-scenario banks
        # (this is what previously made E5b_cbar ~480MB instead of ~50MB).
        base = {
            "benchmark_version": "CBAR-MDVRP-PUBLIC-v1.0",
            "experiment": "E5b-public-backbone-CBAR",
            "source_instance": name,
            "source_reference_type": ref_type,
            "published_bks_distance": published_bks,
            "retained_public_data": asdict(inst),
            "eligibility": "all public depots",
            "emission_parameters": {"theta": 1.0, "rho_empty": 0.9, "rho_full": 1.5},
            "emission_cost": EMISSION_COST,
            "reference_emission": ref_e,
            "reference_demand": ref_r,
            "reference_demand_per_depot": {str(k): v for k, v in per_depot_r0.items()},
            "prices": {"buy": P_BUY, "sell": P_SELL},
            "scenario_seeds": {
                "train": STREAM_BASE["train"] + idx,
                "validation": STREAM_BASE["validation"] + idx,
                "test": STREAM_BASE["test"] + idx,
            },
            "scenarios": {
                "train": correlated_scenarios(inst, TRAIN_N, STREAM_BASE["train"] + idx),
                "validation": correlated_scenarios(inst, VALID_N, STREAM_BASE["validation"] + idx),
                "test": correlated_scenarios(inst, TEST_N, STREAM_BASE["test"] + idx),
            },
        }
        base_path = out / "E5b_cbar" / name / "base.json"
        write_json(base_path, base)
        manifest_rows.append({"experiment": "E5b", "instance": name, "variant": "base", "file": str(base_path.relative_to(out)), "sha256": sha256(base_path), "reference_route": ref_type})

        r0_total = ref_r
        for gamma in BUDGET_FACTORS:
            for tau in CAPACITY_SCALES:
                arcs = [
                    dict(a, capacity=tau * 0.5 * (per_depot_r0.get(a["from"], 0.0) + per_depot_r0.get(a["to"], 0.0)))
                    for a in graph
                ]
                variant = f"g{gamma:.2f}_t{tau:.2f}".replace(".", "p")
                corp_budget = gamma * ref_r
                # Same 45%/160%-of-proportional-allocation rule as the
                # synthetic suites (code/carbon.py, subsec:experimental_setup)
                # -- E5b's own paragraph doesn't restate it, but nothing in
                # it overrides the general budget parameterisation either.
                alloc = {d: corp_budget * r / r0_total for d, r in per_depot_r0.items()} if r0_total > 0 else {}
                bounds = {str(d): [0.45 * a, 1.60 * a] for d, a in alloc.items()}
                overlay = {
                    "benchmark_version": "CBAR-MDVRP-PUBLIC-v1.0",
                    "experiment": "E5b-public-backbone-CBAR",
                    "source_instance": name,
                    "base_file": "base.json",
                    "corporate_budget": corp_budget,
                    "budget_factor": gamma,
                    "depot_budget_allocation": {str(d): a for d, a in alloc.items()},
                    "depot_budget_bounds": bounds,
                    "transfer_capacity_scale": tau,
                    "transfer_arcs": arcs,
                }
                path = out / "E5b_cbar" / name / f"{variant}.json"
                write_json(path, overlay)
                manifest_rows.append({"experiment": "E5b", "instance": name, "variant": variant, "file": str(path.relative_to(out)), "sha256": sha256(path), "reference_route": ref_type})
    with (out / "converted_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=manifest_rows[0].keys()); w.writeheader(); w.writerows(manifest_rows)
    write_json(out / "conversion_summary.json", {
        "parsed_public_instances": len(parsed),
        "e5a_files": sum(r["experiment"] == "E5a" for r in manifest_rows),
        "e5b_files": sum(r["experiment"] == "E5b" for r in manifest_rows),
        "selected_e5b": SELECTED_E5B,
    })


def smoke_test() -> None:
    fixture = """2 2 4 2
0 10
0 10
1 0 1 0 2 1 1 1
2 1 1 0 2 1 1 1
3 9 9 0 2 1 1 1
4 10 9 0 2 1 1 1
5 0 0 0 0 0 0
6 10 10 0 0 0 0
"""
    # tempfile rather than a hardcoded /tmp, which does not exist on Windows.
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "p99"; p.write_text(fixture)
        inst = parse_cordeau_instance(p)
    assert inst.n_customers == 4 and inst.n_depots == 2 and inst.max_vehicles == 2
    s = correlated_scenarios(inst, 5, 123)
    assert len(s) == 5 and abs(sum(x["probability"] for x in s) - 1) < 1e-12
    arcs = mst_plus_nearest(inst)
    assert len(arcs) == 2
    routes, e = nearest_depot_reference(inst)
    assert routes and e > 0
    # Two customers sit beside each depot, each demanding 2, so the reference
    # demand must split 4/4 and total 8.
    r0 = per_depot_reference_demand(inst)
    assert r0 == {1: 4.0, 2: 4.0}, r0
    print(json.dumps({"status": "PASS", "customers": 4, "depots": 2, "scenarios": 5,
                      "arcs": len(arcs), "reference_emission": e,
                      "reference_demand_per_depot": r0}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=Path, help="extracted Cordeau instance directory")
    ap.add_argument("--solutions", type=Path, help="extracted Cordeau solution directory")
    ap.add_argument("--out", type=Path, default=Path("CBAR_MDVRP_public_benchmark_v1/converted"))
    ap.add_argument("--download", action="store_true", help="download public NEO archives before conversion")
    ap.add_argument("--cache", type=Path, default=Path("public_source_cache"))
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    if args.smoke_test:
        smoke_test(); return
    instances, solutions = args.instances, args.solutions
    if args.download:
        args.cache.mkdir(parents=True, exist_ok=True)
        iz, sz = args.cache / "C-mdvrp.zip", args.cache / "C-mdvrp-sol.zip"
        if not iz.exists(): download(INSTANCE_URL, iz)
        if not sz.exists(): download(SOLUTION_URL, sz)
        idir, sdir = args.cache / "instances", args.cache / "solutions"
        extract_zip(iz, idir); extract_zip(sz, sdir)
        instances, solutions = idir, sdir
        write_json(args.out / "source_hashes.json", {"instance_archive_sha256": sha256(iz), "solution_archive_sha256": sha256(sz)})
    if not instances:
        ap.error("provide --instances or use --download")
    build(instances, solutions, args.out)


if __name__ == "__main__":
    main()

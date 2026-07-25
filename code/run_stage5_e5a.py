#!/usr/bin/env python3
"""Stage 4 / E5a: classic MDVRP regression on all 33 public Cordeau
instances (tex, subsec:e5_public_benchmark paragraph "E5a"). Carbon and
uncertainty are disabled; the objective is pure route distance. Reuses
ppbrc.py's ALNS with lambda fixed at 0 (no price signal) via the
cordeau_adapter.PseudoInstance shim, and compares against the published BKS
route cost parsed directly from the raw .res solution file.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import cordeau_adapter
import ppbrc


def run_one(json_path: str, sol_dir: Path, alns_iterations: int, seed: int) -> dict:
    inst = cordeau_adapter.load_e5a(json_path)
    demand = {c["idx"]: c["base_demand"] for c in inst.customers}
    D = [d["idx"] for d in inst.depots]

    t0 = time.time()
    sol = ppbrc.alns_solve(inst, demand, lam={d: 0.0 for d in D}, iterations=alns_iterations, seed=seed)
    runtime = time.time() - t0

    total_distance = sum(sol.depot_metrics(d)[0] for d in D)
    served = sum(len(r) for rs in sol.routes.values() for r in rs)
    vehicles_used = sum(1 for rs in sol.routes.values() for r in rs if r)

    bks_path = sol_dir / f"{inst.name}.res"
    bks = cordeau_adapter.parse_bks_distance(bks_path)
    gap_pct = (total_distance - bks) / bks * 100 if bks else None

    return {
        "instance": inst.name, "n_customers": inst.n_customers, "n_depots": inst.n_depots,
        "distance": total_distance, "bks_distance": bks, "gap_to_bks_pct": gap_pct,
        "served": served, "feasible": served == inst.n_customers,
        "vehicles_used": vehicles_used, "vehicles_available": len(inst.vehicles),
        "runtime": runtime,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\CBAR_MDVRP\data\generated\public_v1\E5a_classic")
    ap.add_argument("--solutions", default=r"D:\CBAR_MDVRP\data\raw\C-mdvrp-sol")
    ap.add_argument("--out", default=r"D:\CBAR_MDVRP\data\results\stage5_e5a\stage5_e5a_results.jsonl")
    ap.add_argument("--alns-iterations", type=int, default=600)
    ap.add_argument("--seed", type=int, default=91001)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.data) / "*.json")))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["instance"])

    with out_path.open("a", encoding="utf-8") as f:
        for path in files:
            name = Path(path).stem
            if name in done:
                print(f"skip {name}")
                continue
            print(f"running {name} ...", flush=True)
            row = run_one(path, Path(args.solutions), args.alns_iterations, args.seed)
            f.write(json.dumps(row) + "\n")
            f.flush()
            gap_str = f"{row['gap_to_bks_pct']:.2f}%" if row["gap_to_bks_pct"] is not None else "n/a (no BKS)"
            print(f"  distance={row['distance']:.2f}  bks={row['bks_distance']}  gap={gap_str}  "
                  f"feasible={row['feasible']}  [{row['runtime']:.0f}s]", flush=True)

    print("done.")


if __name__ == "__main__":
    main()

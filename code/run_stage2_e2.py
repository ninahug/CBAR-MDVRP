#!/usr/bin/env python3
"""Comparison with benchmark algorithms (paper Section 6.3) over the 24
instances of the comparison suite.

Runs all six methods of methods.ALL_METHODS on every instance with each of the
five search seeds the paper fixes, giving 24 x 5 x 6 = 720 runs. The search
effort per run (--alns-iterations, --outer-rounds, --price-rounds) is exposed
so that the grid can be completed within an available time budget; the paper
reports the settings actually used.

The script is resumable -- it skips rows already present in --out -- and
shardable via --shard/--num-shards, so the grid can be split across cores.
Do not run more shards than physical cores: oversubscription slows the grid
non-linearly.

Mechanism-activation diagnostics (paper Table 7) are computed once per
instance, from the PPBRC run at the first seed, because they describe a
pre-specified instance class rather than a per-seed outcome.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from instance_io import load_instance
import methods


def row_key(instance: str, seed: int, method: str) -> str:
    return f"{instance}|{seed}|{method}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\CBAR_MDVRP\data\generated\synthetic_v1\E2")
    ap.add_argument("--out", default=r"D:\CBAR_MDVRP\data\results\stage2_e2\stage2_e2_results.jsonl")
    ap.add_argument("--diagnostics-out", default=r"D:\CBAR_MDVRP\data\results\stage2_e2\stage2_e2_diagnostics.jsonl")
    ap.add_argument("--seeds", type=int, nargs="*", default=methods.FROZEN_SEEDS)
    ap.add_argument("--methods", nargs="*", default=methods.ALL_METHODS)
    ap.add_argument("--outer-rounds", type=int, default=2)
    ap.add_argument("--price-rounds", type=int, default=2)
    ap.add_argument("--alns-iterations", type=int, default=80)
    ap.add_argument("--ph-iterations", type=int, default=5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.data) / "*.json")))

    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add(row_key(row["instance"], row["seed"], row["method"]))

    diag_done = set()
    diag_path = Path(args.diagnostics_out)
    if diag_path.exists():
        for line in diag_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                diag_done.add(json.loads(line)["instance"])

    jobs = []
    for path in files:
        name = Path(path).stem
        for seed in args.seeds:
            for method in args.methods:
                jobs.append((path, name, seed, method))

    with out_path.open("a", encoding="utf-8") as f, diag_path.open("a", encoding="utf-8") as fd:
        cache: dict[str, tuple] = {}
        for idx, (path, name, seed, method) in enumerate(jobs):
            if idx % args.num_shards != args.shard:
                continue
            key = row_key(name, seed, method)
            if key in done:
                continue
            if path not in cache:
                inst = load_instance(path)
                cache = {path: (inst, inst.scenarios("train"))}  # keep only the current instance loaded
            inst, scenarios = cache[path]

            print(f"[shard {args.shard}] running {name} seed={seed} method={method} ...", flush=True)
            t0 = time.time()
            result = methods.run_method(method, inst, scenarios, seed=seed,
                                         outer_rounds=args.outer_rounds, price_rounds=args.price_rounds,
                                         alns_iterations=args.alns_iterations, ph_iterations=args.ph_iterations)
            runtime = time.time() - t0
            row = {
                "instance": name, "n_customers": inst.n_customers, "n_depots": inst.n_depots,
                "seed": seed, "method": method,
                "objective": result["objective"], "distance": result["distance"],
                "recourse_cost": result["recourse_cost"], "runtime": runtime,
            }
            if method == "PH-ALNS":
                row["budget_consensus_residual"] = result["budget_consensus_residual"]
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"  objective={row['objective']:.4f}  runtime={runtime:.1f}s", flush=True)

            if method == "PPBRC" and seed == methods.FROZEN_SEEDS[0] and name not in diag_done:
                diag = methods.mechanism_diagnostics(inst, result)
                diag["instance"] = name
                fd.write(json.dumps(diag) + "\n")
                fd.flush()
                diag_done.add(name)

    print("done.")


if __name__ == "__main__":
    main()

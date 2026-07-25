#!/usr/bin/env python3
"""Stage 3 / E4: interaction of operational and accounting recourse (tex,
Table frozen_data_suites' E4 row + the 2^4 factorial paragraph after it).

Runs PPBRC (full mechanism) and Cost-only coordination (operational
recourse only, no price-guided routing) on all 32 E4 instances so the
paper's interaction analysis can compare "both mechanisms" against
"operational recourse alone" across the factorial design (demand CV,
cross-depot correlation, boundary share, capacity scale) recorded in each
instance's own `factors` block -- no separate design matrix needs to be
carried here, it can be joined from suite_config.build_e4_specs() later.

Resumable (skips rows already in --out) and shardable like E2's harness.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from instance_io import load_instance
import methods


def row_key(instance: str, method: str) -> str:
    return f"{instance}|{method}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\CBAR_MDVRP\data\generated\synthetic_v1\E4")
    ap.add_argument("--out", default=r"D:\CBAR_MDVRP\data\results\stage4_e4\stage4_e4_results.jsonl")
    ap.add_argument("--methods", nargs="*", default=["PPBRC", "Cost-only"])
    ap.add_argument("--seed", type=int, default=91001)
    ap.add_argument("--outer-rounds", type=int, default=3)
    ap.add_argument("--price-rounds", type=int, default=2)
    ap.add_argument("--alns-iterations", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.data) / "*.json")))
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add(row_key(row["instance"], row["method"]))

    jobs = [(path, method) for path in files for method in args.methods]

    with out_path.open("a", encoding="utf-8") as f:
        for idx, (path, method) in enumerate(jobs):
            if idx % args.num_shards != args.shard:
                continue
            inst = load_instance(path)
            key = row_key(inst.name, method)
            if key in done:
                continue
            scenarios = inst.scenarios("train")
            print(f"[shard {args.shard}] {inst.name} method={method} ...", flush=True)
            t0 = time.time()
            result = methods.run_method(method, inst, scenarios, seed=args.seed,
                                         outer_rounds=args.outer_rounds, price_rounds=args.price_rounds,
                                         alns_iterations=args.alns_iterations)
            runtime = time.time() - t0
            diag = methods.mechanism_diagnostics(inst, result)
            row = {
                "instance": inst.name, "method": method, "factors": inst.raw["factors"],
                "objective": result["objective"], "distance": result["distance"],
                "recourse_cost": result["recourse_cost"], "runtime": runtime,
                **diag,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"  objective={row['objective']:.4f}  runtime={runtime:.1f}s", flush=True)

    print("done.")


if __name__ == "__main__":
    main()

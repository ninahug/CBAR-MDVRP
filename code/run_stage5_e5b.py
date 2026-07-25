#!/usr/bin/env python3
"""Stage 4 / E5b: CBAR on public Cordeau backbones (tex, subsec:e5_public_
benchmark paragraph "E5b"), on the 11 selected instances
{p01,p02,p03,p12,p04,p05,p06,p07,p15,p18,p21}.

Runs SAA-ALNS, PH-ALNS, PPBRC-core, PPBRC (the four the tex names for E5b)
via cordeau_adapter.load_e5b + methods.py's existing dispatcher -- no new
routing/algorithm code, only a loader was needed.

Scope note: the frozen protocol crosses 3 budget factors x 3 capacity
scales (9 variants) per instance; all 9 are generated
(data/generated/public_v1/E5b_cbar/<instance>/g*_t*.json) but this first
pass runs only the baseline cell (g1.00_t1.00, i.e. gamma_B=1, tau=1) across
all 11 instances -- the other 8 variants per instance are available for a
later robustness/sensitivity pass without regenerating anything.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cordeau_adapter
import methods

SELECTED_E5B = ["p01", "p02", "p03", "p12", "p04", "p05", "p06", "p07", "p15", "p18", "p21"]


def row_key(instance: str, variant: str, seed: int, method: str) -> str:
    return f"{instance}|{variant}|{seed}|{method}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\CBAR_MDVRP\data\generated\public_v1\E5b_cbar")
    ap.add_argument("--out", default=r"D:\CBAR_MDVRP\data\results\stage5_e5b\stage5_e5b_results.jsonl")
    ap.add_argument("--variants", nargs="*", default=["g1p00_t1p00"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[91001, 91002])
    ap.add_argument("--methods", nargs="*", default=["SAA-ALNS", "PH-ALNS", "PPBRC-core", "PPBRC"])
    ap.add_argument("--outer-rounds", type=int, default=2)
    ap.add_argument("--price-rounds", type=int, default=2)
    ap.add_argument("--alns-iterations", type=int, default=100)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add(row_key(r["instance"], r["variant"], r["seed"], r["method"]))

    with out_path.open("a", encoding="utf-8") as f:
        for name in SELECTED_E5B:
            for variant in args.variants:
                inst = cordeau_adapter.load_e5b(Path(args.data) / name, variant)
                scenarios = inst.scenarios("train")
                for seed in args.seeds:
                    for method in args.methods:
                        key = row_key(name, variant, seed, method)
                        if key in done:
                            continue
                        print(f"running {name}/{variant} seed={seed} method={method} ...", flush=True)
                        t0 = time.time()
                        result = methods.run_method(method, inst, scenarios, seed=seed,
                                                     outer_rounds=args.outer_rounds,
                                                     price_rounds=args.price_rounds,
                                                     alns_iterations=args.alns_iterations)
                        runtime = time.time() - t0
                        row = {
                            "instance": name, "variant": variant, "seed": seed, "method": method,
                            "n_customers": inst.n_customers, "n_depots": inst.n_depots,
                            "objective": result["objective"], "distance": result["distance"],
                            "recourse_cost": result["recourse_cost"], "runtime": runtime,
                        }
                        if method == "PH-ALNS":
                            row["budget_consensus_residual"] = result["budget_consensus_residual"]
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        print(f"  objective={row['objective']:.4f}  runtime={runtime:.1f}s", flush=True)

    print("done.")


if __name__ == "__main__":
    main()

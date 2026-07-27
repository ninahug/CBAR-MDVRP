#!/usr/bin/env python3
"""Paired comparison of PPBRC with and without route recombination.

Recombination is only worth its cost if it improves the exact objective. The
two variants are run on the same instances with the same seeds and identical
search settings, so the paired difference isolates recombination. Because a
recombined plan replaces the incumbent only on a strict improvement of the
exact scenario cost, no run should ever be worse; the question the test
answers is whether it is meaningfully better and at what cost in time.
"""
from __future__ import annotations

import argparse
import glob
import statistics
import sys
import time
from pathlib import Path

from instance_io import load_instance
import ppbrc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\CBAR_MDVRP\data\generated\synthetic_v1\E2")
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*", default=[91001, 91002])
    ap.add_argument("--scenarios", type=int, default=8)
    ap.add_argument("--alns-iterations", type=int, default=60)
    ap.add_argument("--outer-rounds", type=int, default=2)
    ap.add_argument("--price-rounds", type=int, default=2)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.data) / "*.json")))[:args.instances]
    rows = []
    for path in files:
        inst = load_instance(path)
        scenarios = inst.scenarios("train")[:args.scenarios]
        for seed in args.seeds:
            out = {}
            for tag, flag in (("base", False), ("recomb", True)):
                t0 = time.time()
                r = ppbrc.run_ppbrc(
                    inst, scenarios, outer_rounds=args.outer_rounds,
                    price_rounds=args.price_rounds,
                    alns_iterations=args.alns_iterations, seed=seed,
                    recombine_pool=flag)
                out[tag] = (r["objective"], time.time() - t0, r)
            base_obj, base_t, _ = out["base"]
            rec_obj, rec_t, rec_r = out["recomb"]
            denom = abs(base_obj) if abs(base_obj) > 1e-9 else 1.0
            gain = (base_obj - rec_obj) / denom * 100.0
            rows.append((inst.name, seed, base_obj, rec_obj, gain, base_t, rec_t,
                         rec_r["recombined_accepted"], rec_r["recombined_attempts"]))
            print(f"{inst.name:28s} seed={seed}  base={base_obj:10.4f}  "
                  f"recomb={rec_obj:10.4f}  gain={gain:+6.2f}%  "
                  f"time {base_t:5.1f}s -> {rec_t:5.1f}s  "
                  f"accepted {rec_r['recombined_accepted']}/{rec_r['recombined_attempts']}",
                  flush=True)

    gains = [r[4] for r in rows]
    worse = [r for r in rows if r[3] > r[2] + 1e-6]
    better = [r for r in rows if r[3] < r[2] - 1e-6]
    tbase = sum(r[5] for r in rows)
    trec = sum(r[6] for r in rows)

    print("\n--- paired summary over %d runs ---" % len(rows))
    print(f"mean improvement   : {statistics.mean(gains):+.3f}%")
    print(f"median improvement : {statistics.median(gains):+.3f}%")
    print(f"best / worst       : {max(gains):+.3f}% / {min(gains):+.3f}%")
    print(f"better / equal / worse : {len(better)} / {len(rows)-len(better)-len(worse)} / {len(worse)}")
    print(f"total time         : {tbase:.1f}s -> {trec:.1f}s  ({trec/max(tbase,1e-9):.2f}x)")

    if worse:
        print("\nFAIL: recombination made some runs worse, which the acceptance "
              "rule should have prevented:")
        for r in worse:
            print("   ", r[0], r[1], r[2], "->", r[3])
        return 1
    print("\nNo run was made worse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

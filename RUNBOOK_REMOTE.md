# Running the remaining CBAR-MDVRP experiments on another machine

## 1. Prerequisites

- Python 3.11+ (3.13 used here)
- `pip install pulp numpy` (that's the entire external dependency list;
  PuLP bundles its own CBC binary, no separate solver install needed)

## 2. Get the code

```bash
git clone https://github.com/ninahug/CBAR-MDVRP.git
cd CBAR-MDVRP
```

## 3. Get the data

`data/` is gitignored (regenerable, and the E5 raw files + generated
suites are ~180MB together — not worth versioning). Two pieces needed:

**3a. Raw Cordeau benchmark files** (394KB, only source needed — everything
else is generated from this). Either copy `data/raw/` from this machine, or
fetch fresh:
```bash
python gpt_test/build_cordeau_cbar_benchmark.py --download --out data/generated/public_v1
# downloads C-mdvrp.zip / C-mdvrp-sol.zip from neo.lcc.uma.es into
# public_source_cache/, extracts to public_source_cache/instances,solutions
```
If you copy `data/raw/` from this machine instead, regenerate with:
```bash
python gpt_test/build_cordeau_cbar_benchmark.py --instances data/raw/C-mdvrp --solutions data/raw/C-mdvrp-sol --out data/generated/public_v1
```

**3b. The 100 synthetic instances** (tuning/E1/E2/E3/E4) — always generated,
never downloaded:
```bash
cd code
python build_data_release.py          # writes data/generated/synthetic_v1/
python validate_release.py            # should print "all checks PASSED"
```

**3c. (Recommended) Bring over partial progress** so you don't redo work
already done on this machine. Copy `data/results/` from here — every
harness below is resumable and skips rows already present in its `--out`
file. Current state as of this handoff:

| file | rows done |
|---|---|
| `results/stage1_e1/stage1_e1_results.jsonl` | 21 / 36 |
| `results/stage2_e2/stage2_e2_results.jsonl` | 41 / 216 |
| `results/stage3_e3/stage3_e3_results.jsonl` | 0 / 8 |
| `results/stage4_e4/stage4_e4_results.jsonl` | 4 / 64 |
| `results/stage5_e5a/stage5_e5a_results.jsonl` | 6 / 33 |
| `results/stage5_e5b/stage5_e5b_results.jsonl` | 1 / 88 |

## 4. Run each stage

Run `nproc` (or `python -c "import os;print(os.cpu_count())"` on Windows)
first and set `N` below to that core count. **Do not launch more parallel
processes than cores** — oversubscribing badly hurt throughput last time
(non-linearly, not just proportionally).

```bash
cd code

# Stage 1 -- E1 (36 instances). Run ALONE if you want the exact-MILP timing
# to be meaningful (CBC's time limit is wall-clock based); otherwise fine
# to run alongside the others since integer-feasibility is checked
# explicitly now (a fractional CBC relaxation can no longer be mislabeled
# "proven optimal" -- see exact_model.py's integer_feasible check).
python run_stage1_e1.py

# Stage 2 -- E2 (216 runs). Shard across N processes, one per core, e.g. N=8:
python run_stage2_e2.py --num-shards 8 --shard 0   # ... --shard 1 ... up to N-1

# Stage 3 -- E3 (8 instances, already tuned to a tractable eval-subsample=10 /
# reduced-iteration setting after the first attempt implied 10+ hours):
python run_stage3_e3.py

# Stage 4 -- E4 (64 runs), shard similarly:
python run_stage4_e4.py --num-shards N --shard 0   # ... up to N-1

# Stage 4 -- E5a (33 instances, classic MDVRP vs BKS):
python run_stage5_e5a.py

# Stage 4 -- E5b (88 runs at the baseline g1.00_t1.00 grid cell; the other
# 8 budget/capacity variants per instance are already generated under
# data/generated/public_v1/E5b_cbar/<instance>/g*_t*.json for a later
# robustness pass via --variants):
python run_stage5_e5b.py
```

Total remaining work across all six, run in parallel across ~8 cores, was
last estimated at roughly 4-6 hours wall-clock, with E5b (single process,
no sharding built in yet -- see below) as the long pole.

## 5. If you have spare cores, shard E5a/E5b too

`run_stage5_e5a.py` and `run_stage5_e5b.py` don't currently take
`--shard`/`--num-shards` (E1 and E3 don't either, by design — E1 for clean
timing, E3 because 8 instances barely benefits from sharding). If the other
machine has many cores free after E2/E4 finish, the simplest way to
parallelize E5a/E5b further is to run multiple copies with disjoint
`--data`/instance subsets (or ask me to add `--shard` to those two scripts
— it's a small, mechanical change following the same pattern as
`run_stage2_e2.py`).

## 6. When done

Copy `data/results/` back (or just the six `*.jsonl` files) so Stage 5
(writing everything into the tex) can proceed from the merged results.

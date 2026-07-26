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

**3b. The 102 synthetic instances** (8 tuning, 30 verification, 24
comparison, 8 stochastic-value, 32 sensitivity) — always generated, never
downloaded:
```bash
cd code
python build_data_release.py          # writes data/generated/synthetic_v1/
python validate_release.py            # should print "all checks PASSED"
```

**3c. Nothing to carry over — start from empty results.** The instance
suites were resized to round counts (see the table below), so every earlier
partial result was computed against instances that no longer exist and has
been archived under `data/results_stale_pre_resize/`. Do **not** copy that
directory into `data/results/`: the harnesses resume by instance name, and
stale rows would be silently accepted as done. Start with `data/results/`
empty.

Work to be done, per stage:

| stage | runs |
|---|---|
| verification (`run_stage1_e1.py`) | 30 instances |
| comparison (`run_stage2_e2.py`) | 24 instances x 5 seeds x 6 methods = 720 |
| stochastic value (`run_stage3_e3.py`) | 8 instances |
| sensitivity (`run_stage4_e4.py`) | 32 instances x 4 recourse regimes = 128 solves |
| public benchmark E5a (`run_stage5_e5a.py`) | 33 instances |
| public benchmark E5b (`run_stage5_e5b.py`) | 11 instances x 5 seeds x 4 methods = 220 |

## 4. Run each stage

Run `nproc` (or `python -c "import os;print(os.cpu_count())"` on Windows)
first and set `N` below to that core count. **Do not launch more parallel
processes than cores** — oversubscribing badly hurt throughput last time
(non-linearly, not just proportionally).

```bash
cd code

# Verification (30 instances). Run ALONE if you want the exact-MILP timing
# to be meaningful (CBC's time limit is wall-clock based); otherwise fine
# to run alongside the others since integer-feasibility is checked
# explicitly now (a fractional CBC relaxation can no longer be mislabeled
# "proven optimal" -- see exact_model.py's integer_feasible check).
python run_stage1_e1.py

# Comparison (720 runs). Shard across N processes, one per core, e.g. N=8:
python run_stage2_e2.py --num-shards 8 --shard 0   # ... --shard 1 ... up to N-1

# Stochastic value (8 instances; already tuned to a tractable
# eval-subsample=10 / reduced-iteration setting after the first attempt
# implied 10+ hours):
python run_stage3_e3.py

# Sensitivity (32 instances, each solved under 4 recourse regimes), shard similarly:
python run_stage4_e4.py --num-shards N --shard 0   # ... up to N-1

# Public benchmark E5a (33 instances, classic MDVRP vs BKS):
python run_stage5_e5a.py

# Public benchmark E5b (220 runs at the baseline g1.00_t1.00 grid cell; the other
# 8 budget/capacity variants per instance are already generated under
# data/generated/public_v1/E5b_cbar/<instance>/g*_t*.json for a later
# robustness pass via --variants):
python run_stage5_e5b.py
```

Total work across all six, run in parallel across ~8 cores, is on the order
of 20-30 hours wall-clock. The sensitivity suite is heavier than its instance
count suggests, because each of its 32 instances is solved four times, once
per recourse regime. The seed count now follows the paper's protocol of
five search seeds, which puts the comparison suite at 720 runs and E5b at 220,
and nothing is carried over from earlier partial runs. E5b is the long pole,
being a single process with no sharding built in; if the machine has cores to
spare, see the next section.

## 5. If you have spare cores, shard E5a/E5b too

`run_stage5_e5a.py` and `run_stage5_e5b.py` don't currently take
`--shard`/`--num-shards` (the verification and stochastic-value harnesses
don't either, by design — the former for clean exact-solver timing, the
latter because 8 instances barely benefits). If the other machine has many
cores free after the comparison and sensitivity suites finish, the simplest way to
parallelize E5a/E5b further is to run multiple copies with disjoint
`--data`/instance subsets (or ask me to add `--shard` to those two scripts
— it's a small, mechanical change following the same pattern as
`run_stage2_e2.py`).

## 6. When done

Copy `data/results/` back (or just the six `*.jsonl` files) so Stage 5
(writing everything into the tex) can proceed from the merged results.

# Running the remaining IAR-MDVRP experiments on another machine

The repository and directory are still named `CBAR-MDVRP` for historical
reasons. The model is no longer about carbon: it is the **inventory allocation
and recourse MDVRP (IAR-MDVRP)**, in which stock is positioned across depots
before demand is known and rebalanced afterwards by reassigning customers, by
transshipping stock between depots, or by emergency supply and salvage.

## 0. Read this first if you ran anything before 2026-07-27

**Every previously generated instance and every previously computed result is
invalid.** This is not a resizing issue like the last round; the meaning of the
model changed. The quantity charged to a depot's account used to be the
load-dependent emission of its routes and is now the demand assigned to it, the
external prices went from 9/2 to 2.0/0.5, and the reference scale went from a
constructed angle-sweep emission to the base demand homed at each depot.

Concretely, before doing anything else:

```bash
rm -rf data/generated/synthetic_v1 data/generated/public_v1 data/results
```

Do **not** copy `data/results_stale_pre_resize/` (or any other archived
results) into `data/results/`. The harnesses resume by instance name, so stale
rows are silently accepted as finished and would quietly poison the tables.

## 1. Prerequisites

- Python 3.11+ (3.13 used here)
- `pip install pulp numpy` — that is the entire external dependency list.
  PuLP bundles its own CBC binary, so no separate solver install is needed.

## 2. Get the code

```bash
git clone https://github.com/ninahug/CBAR-MDVRP.git
cd CBAR-MDVRP
```

## 3. Generate the data

`data/` is gitignored: it is fully regenerable, and the raw E5 files plus the
generated suites come to roughly 180MB. Both suites below **must** be generated
fresh — see section 0.

**3a. The 102 synthetic instances** (8 tuning, 30 verification, 24 comparison,
8 stochastic-value, 32 sensitivity):

```bash
cd code
python build_data_release.py          # writes data/generated/synthetic_v1/
python validate_release.py            # must print "all checks PASSED"
cd ..
```

**3b. The public Cordeau suites.** The only source needed is the raw Cordeau
archive (394KB); everything else is derived from it. Either copy `data/raw/`
from the origin machine and run:

```bash
python gpt_test/build_cordeau_cbar_benchmark.py \
    --instances data/raw/C-mdvrp --solutions data/raw/C-mdvrp-sol \
    --out data/generated/public_v1
```

or fetch the archives fresh:

```bash
python gpt_test/build_cordeau_cbar_benchmark.py --download --out data/generated/public_v1
# pulls C-mdvrp.zip / C-mdvrp-sol.zip from neo.lcc.uma.es into
# public_source_cache/ and extracts to public_source_cache/{instances,solutions}
```

Both commands are run from the repository root, not from `code/`.

Confirm the E5b overlay carries the inventory semantics rather than the old
carbon ones — if `emission_cost` is missing, the data is stale and every E5b
run would silently price fuel at zero:

```bash
python -c "import json; b=json.load(open('data/generated/public_v1/E5b_cbar/p01/base.json')); \
print(b['emission_cost'], b['prices'], round(b['reference_demand'],1))"
# expect: 0.1 {'buy': 2.0, 'sell': 0.5} 777.0
```

## 4. Sanity check before committing 20-30 hours

```bash
cd code
python run_stage0_smoke_test.py       # a few minutes; must end "ALL CHECKS PASSED"
```

It solves a tiny instance exactly, checks recourse strong duality and
complementary slackness, verifies the institutional ordering
`Z_pool <= Z_network <= Z_independent`, and checks `VSS >= 0`. If this fails,
stop — nothing downstream will be meaningful.

Work to be done, per stage:

| stage | runs |
|---|---|
| verification (`run_stage1_e1.py`) | 30 instances |
| comparison (`run_stage2_e2.py`) | 24 instances x 5 seeds x 6 methods = 720 |
| stochastic value (`run_stage3_e3.py`) | 8 instances |
| sensitivity (`run_stage4_e4.py`) | 32 instances x 4 recourse regimes = 128 solves |
| public benchmark E5a (`run_stage5_e5a.py`) | 33 instances |
| public benchmark E5b (`run_stage5_e5b.py`) | 11 instances x 5 seeds x 4 methods = 220 |

## 5. Run each stage

Run `nproc` (or `python -c "import os;print(os.cpu_count())"` on Windows)
first and set `N` below to that core count. **Do not launch more parallel
processes than cores** — oversubscribing hurt throughput badly last time, and
non-linearly rather than proportionally.

```bash
cd code

# Verification (30 instances). Run ALONE if you want the exact-MILP timings to
# be meaningful, since CBC's time limit is wall-clock based. Otherwise it is
# safe to run alongside the others: integer feasibility is now checked
# explicitly (exact_model.py:207), so a fractional CBC relaxation can no
# longer be mislabelled "proven optimal".
python run_stage1_e1.py

# Comparison (720 runs). Shardable; one shard per core, e.g. N=8:
python run_stage2_e2.py --num-shards 8 --shard 0   # ... --shard 1 ... up to N-1

# Stochastic value (8 instances; already tuned to a tractable
# eval-subsample=10 / reduced-iteration setting after the first attempt
# implied 10+ hours):
python run_stage3_e3.py

# Sensitivity (32 instances, each solved under 4 recourse regimes). Shardable:
python run_stage4_e4.py --num-shards N --shard 0   # ... up to N-1

# Public benchmark E5a (33 instances, classic MDVRP against the published BKS):
python run_stage5_e5a.py

# Public benchmark E5b (220 runs at the baseline g1p00_t1p00 cell; the other
# eight budget/capacity variants per instance are generated under
# data/generated/public_v1/E5b_cbar/<instance>/g*_t*.json and can be run later
# via --variants for the robustness pass):
python run_stage5_e5b.py
```

Total work across the six stages, parallelised across ~8 cores, is on the order
of 20-30 hours wall-clock. The sensitivity suite is heavier than its instance
count suggests because each of its 32 instances is solved four times, once per
recourse regime. E5b is the long pole, being a single process with no sharding
built in.

## 6. If you have spare cores, shard E5a/E5b too

Only `run_stage2_e2.py` and `run_stage4_e4.py` accept `--shard`/`--num-shards`.
The verification and stochastic-value harnesses deliberately do not: the former
so exact-solver timings stay clean, the latter because 8 instances barely
benefit. E5a and E5b simply never had it added.

If the machine has cores free once the comparison and sensitivity suites
finish, the simplest way to parallelise E5a/E5b further is to run several
copies over disjoint instance subsets via `--data`. Adding `--shard` to those
two scripts is a small mechanical change following the `run_stage2_e2.py`
pattern — ask if it is worth doing.

## 7. When done

Copy `data/results/` back (or just the six `*.jsonl` files) so the results can
be written into the tex. The paper's result tables are currently empty
placeholders waiting on exactly these runs; everything else in it — model,
theory, method, parameter tables — is finished.

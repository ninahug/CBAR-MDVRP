# PPBRC public benchmark extension - frozen protocol report

## Decision

The final paper will not rely only on generated instances. A separate E5 public
benchmark layer is added using the Cordeau MDVRP suite.

## E5a: unmodified classic MDVRP regression

- Run the routing engine on all 33 public Cordeau MDVRP instances.
- Disable uncertainty, emissions, carbon budgets, internal transfers, and
  external trading.
- Restore the original distance/duration objective.
- Compare best and mean route cost against published BKS values.
- Audit vehicle count, capacity, route duration, customer coverage, and
  feasibility.

This test validates the routing substrate. It does not validate the carbon
model or the stochastic first-stage decision.

## E5b: full CBAR on public backbones

The complete augmented experiment is frozen on:

`p01, p02, p03, p12, p04, p05, p06, p07, p15, p18, p21`.

These instances span 50-360 customers and 2-9 depots. The following original
fields are retained exactly:

- customer and depot coordinates;
- mean customer demands;
- vehicle capacities;
- route-duration limits;
- maximum vehicle counts.

The paper-specific layers are added by a published conversion protocol:

- 50 training, 200 validation, and 800 sealed test scenarios;
- mean-corrected correlated lognormal demand multipliers;
- the paper's load-dependent emission model;
- budget factors 0.90, 1.00, and 1.10;
- transfer-capacity scales 0.60, 1.00, and 1.40;
- a deterministic depot-distance transfer network;
- identical search seeds and time budgets for SAA-ALNS, PH-ALNS, PPBRC-core,
  and PPBRC.

The original MDVRP BKS is **not** used as a lower bound for the augmented CBAR
objective. E5b compares algorithms only on identical transformed instances.

## Reproducibility controls

The adapter records source archive hashes, converted-file hashes, scenario
seeds, and whether the reference emission came from a parsed public BKS route
or the explicitly flagged fallback heuristic. No public node or depot may be
deleted, moved, or selected after observing algorithm results.

## Current validation status

- Python syntax check: PASS.
- Cordeau-format parser and scenario-generator smoke test: PASS.
- TeX compilation: PASS.
- PDF render inspection: PASS.
- Raw public archive materialization: not completed in this runtime.
- E5a/E5b numerical results: not yet run and must remain unreported.

The next executable step is to place the official instance and solution
archives beside the adapter (or provide extracted folders), run the conversion,
verify source hashes, and then launch the frozen E5a and E5b experiments.

# CBAR-MDVRP public benchmark extension v1.0

This package defines the frozen public-benchmark layer for the CBAR-MDVRP paper.
It does **not** redistribute the original Cordeau files. Download the public
instance and solution archives from the NEO MDVRP repository, then run the
adapter script.

## Two separate validation tasks

1. **E5a - classic MDVRP regression.** Disable carbon budgets, transfers, and
   uncertainty. The objective becomes the original MDVRP route distance. Run
   the routing engine on all 33 public instances and compare with the published
   best-known solutions. This validates the route representation and search
   substrate only.
2. **E5b - public-backbone CBAR validation.** Retain the public coordinates,
   depots, demands, vehicle capacity, route-duration limit, and vehicle limit.
   Add the paper's stochastic-demand and carbon-account layers by the frozen
   deterministic rules in `public_benchmark_protocol.json`. Compare PPBRC with
   SAA-ALNS and PH-ALNS on the same augmented files. Do not compare the CBAR
   objective directly with the original MDVRP BKS.

## Selected full-CBAR instances

The full augmented experiment uses p01, p02, p03, p12, p04, p05, p06, p07,
p15, p18, and p21. They span 50-360 customers and 2-9 depots. The unmodified
routing regression uses all 33 Cordeau instances.

## Reproducibility

The adapter records source archive SHA-256 hashes, each converted instance
hash, all scenario seeds, and the BKS-route parsing status. If the source
solution file cannot be parsed, the fallback reference route is explicitly
flagged and must not be presented as a public BKS route.

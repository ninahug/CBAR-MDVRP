"""Suite configuration for the numerical-experiment dataset.

Transcribed from the "Experimental setup" subsection and the instance-suite
table of t-mdvrp_CBAR_PPBRC_public_benchmark_protocol.tex. Every suite is
produced by the same generation procedure (geometry.py + demand.py +
carbon.py + network.py); the suites differ only in the scale and factor
settings supplied here.

Instance counts are deliberately round (10/30/30/10) except for the
sensitivity suite, whose size is fixed by its design: a 2^4 factorial has
16 cells, replicated twice for 32 runs. Rounding that to 30 would unbalance
the design and destroy the factor attribution the suite exists to provide,
so it is left at 32 and described in the paper as "16 cells x 2 replicates".

Values marked ASSUMPTION are not given a concrete number in the tex prose
and are documented calibration choices, recorded in every generated
instance's "factors" block so they stay auditable.

seed_index is assigned once, globally, over the concatenated list returned
by all_specs() -- this is what "the same instance index is added to each
[seed] base" means in the "Demand streams" paragraph.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_DEMAND_CV_COMMON = 0.25   # ASSUMPTION baseline depot-common cv (matches E5b)
DEFAULT_DEMAND_CV_IDIO = 0.10     # ASSUMPTION baseline idiosyncratic cv (matches E5b)
DEFAULT_CROSS_DEPOT_CORR = 0.25   # ASSUMPTION baseline equicorrelation (matches E5b)
DEFAULT_BOUNDARY_SHARE = 0.20     # ASSUMPTION baseline boundary-customer share
DEFAULT_BUDGET_FACTOR = 1.00
DEFAULT_CAPACITY_SCALE = 1.00

# Shared size grid used by the tuning, comparison, and stochastic-value
# suites, so instance scale is directly comparable across them. The
# 50/75/100 sizes are the conventional medium-instance grid in this
# literature (and the sizes this paper's comparison suite originally used);
# 20 is added as a small case that the exact solver can still reach.
SIZE_GRID = [20, 50, 75, 100]
DEPOT_GRID = [3, 5]

# Verification suite: small enough for the exact MILP to be meaningful.
VERIFICATION_CUSTOMERS = [10, 20, 30]
VERIFICATION_DEPOTS = [2, 3]
VERIFICATION_SCENARIOS = [2, 4, 6, 8, 10]

# The paper names three comparison strata ("tight-budget/low-capacity",
# "balanced", "loose-budget/high-capacity") without numeric (gamma_B, tau)
# pairs. ASSUMPTION: reuse the grid endpoints/midpoint the public-benchmark
# overlay freezes explicitly (budget factors {0.90,1.00,1.10}, capacity
# scales {0.60,1.00,1.40}), paired monotonically so "tight" means both
# scarce budget and scarce transfer capacity.
COMPARISON_STRATA = [
    ("tight_low_capacity", 0.90, 0.60),
    ("balanced", 1.00, 1.00),
    ("loose_high_capacity", 1.10, 1.40),
]

# Sensitivity suite: the 2^4 factor levels are stated explicitly in the tex.
SENSITIVITY_CUSTOMERS = 60
SENSITIVITY_DEPOTS = 3
SENSITIVITY_DEMAND_CV = [0.15, 0.35]
SENSITIVITY_CORRELATION = [-0.25, 0.55]
SENSITIVITY_BOUNDARY_SHARE = [0.10, 0.30]
SENSITIVITY_CAPACITY_SCALE = [0.60, 1.40]
SENSITIVITY_REPLICATES = 2


@dataclass
class SuiteInstanceSpec:
    suite: str
    name: str
    n_customers: int
    n_depots: int
    train_scenarios: int
    eval_banks: dict = field(default_factory=dict)   # {"validation": n, "test": n, "shift": n}
    budget_factor: float = DEFAULT_BUDGET_FACTOR
    capacity_scale: float = DEFAULT_CAPACITY_SCALE
    demand_cv_common: float = DEFAULT_DEMAND_CV_COMMON
    demand_cv_idio: float = DEFAULT_DEMAND_CV_IDIO
    cross_depot_corr: float = DEFAULT_CROSS_DEPOT_CORR
    boundary_share: float = DEFAULT_BOUNDARY_SHARE
    nested_prefixes: list | None = None   # stochastic-value suite: e.g. [20, 50, 100]
    seed_index: int = 0                   # assigned in all_specs()
    geometry_group: str | None = None     # specs sharing this key share one
                                           # geometry/demand/scenario seed --
                                           # used so the comparison suite's
                                           # three carbon-network strata
                                           # compare the *same* instance under
                                           # different (budget_factor,
                                           # capacity_scale) rather than three
                                           # independently re-randomised
                                           # instances


def build_tuning_specs() -> list[SuiteInstanceSpec]:
    """10 instances: the shared size grid crossed with the depot grid.
    Used only to fix neighbourhood limits, stopping rules, and wall-clock
    budgets; excluded from every reported performance comparison."""
    return [
        SuiteInstanceSpec("tuning", f"tuning_c{n_c}_d{n_d}", n_c, n_d,
                           train_scenarios=20, eval_banks={"validation": 100, "test": 200})
        for n_c in SIZE_GRID for n_d in DEPOT_GRID
    ]


def build_verification_specs() -> list[SuiteInstanceSpec]:
    """30 instances: 3 customer counts x 2 depot counts x 5 scenario counts.
    Sizes are held small because every instance is solved by the exact
    deterministic-equivalent MILP as ground truth. No held-out banks: the
    finite scenario set is solved exactly."""
    return [
        SuiteInstanceSpec("E1", f"e1_c{n_c}_d{n_d}_w{n_w}", n_c, n_d, train_scenarios=n_w)
        for n_c in VERIFICATION_CUSTOMERS
        for n_d in VERIFICATION_DEPOTS
        for n_w in VERIFICATION_SCENARIOS
    ]


def build_comparison_specs() -> list[SuiteInstanceSpec]:
    """30 instances: 5 customer counts x 2 depot counts x 3 carbon-network
    strata. The three strata within a size class share one geometry_group so
    they are the *same* customers/depots/demand/scenarios under three
    different (budget_factor, capacity_scale) overlays -- otherwise a stratum
    comparison would be confounded by an independently re-randomised
    instance."""
    specs = []
    for n_c in SIZE_GRID:
        for n_d in DEPOT_GRID:
            group = f"e2_c{n_c}_d{n_d}"
            for label, gamma, tau in COMPARISON_STRATA:
                specs.append(SuiteInstanceSpec(
                    "E2", f"e2_c{n_c}_d{n_d}_{label}", n_c, n_d,
                    train_scenarios=20, budget_factor=gamma, capacity_scale=tau,
                    geometry_group=group))
    return specs


def build_stochastic_value_specs() -> list[SuiteInstanceSpec]:
    """10 instances: the shared size grid crossed with the depot grid.
    Training samples of 20/50/100 are nested prefixes of one 100-scenario
    bank, so performance changes are attributable to sample size rather than
    to different random draws."""
    return [
        SuiteInstanceSpec("E3", f"e3_c{n_c}_d{n_d}", n_c, n_d,
                           train_scenarios=100, nested_prefixes=[20, 50, 100],
                           eval_banks={"validation": 200, "test": 800, "shift": 200})
        for n_c in SIZE_GRID for n_d in DEPOT_GRID
    ]


def build_sensitivity_specs() -> list[SuiteInstanceSpec]:
    """32 instances: a replicated 2^4 factorial over demand coefficient of
    variation, cross-depot correlation, boundary-customer share, and
    transfer-capacity scale (16 cells x 2 replicates). Instance size is held
    fixed so the design attributes performance differences to the four
    manipulated factors rather than to incidental geometry changes.

    ASSUMPTION: the single "demand coefficient of variation" factor sets the
    depot-common cv; the idiosyncratic cv stays at its baseline, since the
    tex names only one demand-cv factor, not two."""
    specs = []
    i = 0
    for cv in SENSITIVITY_DEMAND_CV:
        for corr in SENSITIVITY_CORRELATION:
            for bshare in SENSITIVITY_BOUNDARY_SHARE:
                for tau in SENSITIVITY_CAPACITY_SCALE:
                    for _rep in range(SENSITIVITY_REPLICATES):
                        i += 1
                        specs.append(SuiteInstanceSpec(
                            "E4", f"e4_{i:02d}", SENSITIVITY_CUSTOMERS, SENSITIVITY_DEPOTS,
                            train_scenarios=50,
                            eval_banks={"validation": 100, "test": 400},
                            capacity_scale=tau, demand_cv_common=cv,
                            cross_depot_corr=corr, boundary_share=bshare))
    return specs


def all_specs() -> list[SuiteInstanceSpec]:
    specs = (build_tuning_specs() + build_verification_specs() + build_comparison_specs()
             + build_stochastic_value_specs() + build_sensitivity_specs())
    next_index = 1
    group_seed: dict[str, int] = {}
    for spec in specs:
        if spec.geometry_group is not None:
            if spec.geometry_group not in group_seed:
                group_seed[spec.geometry_group] = next_index
                next_index += 1
            spec.seed_index = group_seed[spec.geometry_group]
        else:
            spec.seed_index = next_index
            next_index += 1
    return specs

"""Frozen suite configuration table for CBAR-MDVRP-DATA-v1.0.

Transcribed from Table "frozen_data_suites" and the E1-E4 paragraphs of
t-mdvrp_CBAR_PPBRC_public_benchmark_protocol.tex. Sizes, scenario counts, and
factor grids that the tex states explicitly are reproduced exactly (see the
comment above each builder). Values marked ASSUMPTION are not given a
concrete number in the tex prose and are documented choices, recorded in
every generated instance's "factors" block so they stay auditable.

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

# tex names three E2 strata ("tight-budget/low-capacity", "balanced",
# "loose-budget/high-capacity") without numeric (gamma_B, tau) pairs.
# ASSUMPTION: reuse the grid endpoints/midpoint E5b freezes explicitly
# (budget factors {0.90,1.00,1.10}, capacity scales {0.60,1.00,1.40}),
# paired monotonically so "tight" means both scarce budget and scarce
# transfer capacity.
E2_STRATA = [
    ("tight_low_capacity", 0.90, 0.60),
    ("balanced", 1.00, 1.00),
    ("loose_high_capacity", 1.10, 1.40),
]


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
    nested_prefixes: list | None = None   # E3 only: e.g. [20, 50, 100]
    seed_index: int = 0                   # assigned in all_specs()
    geometry_group: str | None = None     # specs sharing this key share one
                                           # geometry/demand/scenario seed --
                                           # used so E2's three carbon-network
                                           # strata compare the *same*
                                           # instance under different
                                           # (budget_factor, capacity_scale)
                                           # rather than three independently
                                           # re-randomised instances


def build_tuning_specs() -> list[SuiteInstanceSpec]:
    # tex, Table frozen_data_suites: "6 | 40, 60 | 3, 5 | 20 | 100 val + 200 test".
    # ASSUMPTION: the 2x2 (customers, depots) grid does not itself reach 6, so
    # two of the four combinations are replicated with an independent
    # geometry seed to reach the tabulated count of 6.
    combos = [(40, 3), (40, 5), (60, 3), (60, 5), (40, 3), (60, 5)]
    return [
        SuiteInstanceSpec("tuning", f"tuning_{i + 1:02d}", n_c, n_d,
                           train_scenarios=20, eval_banks={"validation": 100, "test": 200})
        for i, (n_c, n_d) in enumerate(combos)
    ]


def build_e1_specs() -> list[SuiteInstanceSpec]:
    # tex, subsec:e1_verification: |C| in {10,15,20}, |D| in {2,3},
    # |W| in {3,5,10}, "two independently generated geometries per size
    # class (36 instances in total)". No train/validation/test split: E1
    # solves the full finite scenario set exactly against a MILP.
    specs = []
    for n_c in (10, 15, 20):
        for n_d in (2, 3):
            for n_w in (3, 5, 10):
                for geom in (1, 2):
                    specs.append(SuiteInstanceSpec(
                        "E1", f"e1_c{n_c}_d{n_d}_w{n_w}_g{geom}", n_c, n_d,
                        train_scenarios=n_w))
    return specs


def build_e2_specs() -> list[SuiteInstanceSpec]:
    # tex, subsec:e2_algorithm: |C| in {50,75,100}, |D| in {3,5}, "three
    # pre-specified carbon-network strata per size class" -> 18 instances,
    # 20 training scenarios each, no held-out banks (E2 evaluates trained
    # solutions directly, unlike E3's train/validation/sealed-test design).
    # The three strata within a size class share one geometry_group so they
    # are the *same* customers/depots/demand/scenarios under three different
    # (budget_factor, capacity_scale) overlays -- otherwise a stratum
    # comparison would be confounded by an independently re-randomised
    # instance, which is exactly the kind of ex-post confound the tex's
    # "mechanism diagnostics ... not an ex-post selection rule" language is
    # trying to keep out.
    specs = []
    for n_c in (50, 75, 100):
        for n_d in (3, 5):
            group = f"e2_c{n_c}_d{n_d}"
            for label, gamma, tau in E2_STRATA:
                specs.append(SuiteInstanceSpec(
                    "E2", f"e2_c{n_c}_d{n_d}_{label}", n_c, n_d,
                    train_scenarios=20, budget_factor=gamma, capacity_scale=tau,
                    geometry_group=group))
    return specs


def build_e3_specs() -> list[SuiteInstanceSpec]:
    # tex, subsec:experimental_setup + Table: |C| in {50,100}, |D| in {3,5},
    # nested 20/50/100 training prefixes of one 100-scenario bank, 200
    # validation + 800 sealed test + 200 distribution-shift scenarios.
    # ASSUMPTION: two independent geometries per (customers, depots)
    # combination reach the tabulated count of 8 (2x2x2).
    specs = []
    for n_c in (50, 100):
        for n_d in (3, 5):
            for geom in (1, 2):
                specs.append(SuiteInstanceSpec(
                    "E3", f"e3_c{n_c}_d{n_d}_g{geom}", n_c, n_d,
                    train_scenarios=100, nested_prefixes=[20, 50, 100],
                    eval_banks={"validation": 200, "test": 800, "shift": 200}))
    return specs


def build_e4_specs() -> list[SuiteInstanceSpec]:
    # tex, subsec:e4_recourse_interaction (paragraph after Table):
    # "replicated 2^4 design over demand coefficient of variation (0.15 or
    # 0.35), cross-depot correlation (-0.25 or 0.55), boundary-customer
    # share (0.10 or 0.30), and transfer-capacity scale (0.60 or 1.40)."
    # Fixed size |C|=60, |D|=3; 16 cells x 2 replicate seeds = 32 instances.
    # ASSUMPTION: the single "demand coefficient of variation" factor sets
    # the depot-common cv (sigma); the idiosyncratic cv stays at the
    # DEFAULT_DEMAND_CV_IDIO baseline, since the tex names only one demand-cv
    # factor, not two.
    specs = []
    i = 0
    for cv in (0.15, 0.35):
        for corr in (-0.25, 0.55):
            for bshare in (0.10, 0.30):
                for tau in (0.60, 1.40):
                    for rep in (1, 2):
                        i += 1
                        specs.append(SuiteInstanceSpec(
                            "E4", f"e4_{i:02d}", 60, 3, train_scenarios=50,
                            eval_banks={"validation": 100, "test": 400},
                            capacity_scale=tau, demand_cv_common=cv,
                            cross_depot_corr=corr, boundary_share=bshare))
    return specs


def all_specs() -> list[SuiteInstanceSpec]:
    specs = (build_tuning_specs() + build_e1_specs() + build_e2_specs()
             + build_e3_specs() + build_e4_specs())
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

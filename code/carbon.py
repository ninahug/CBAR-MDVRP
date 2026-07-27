"""Reference demand, the stock-budget parameterisation, and the fuel model.

The module name is historical: it dates from the carbon version of this model
and is kept only so that import paths stay stable. Nothing here is about
carbon.

Two distinct quantities live in this file.

The *reference demand* R^0_d is the total base demand of the customers homed
at depot d (tex, appendix "Stock parameters and transfer network"). It is the
scale for the stock budget, the depot positioning bounds, and the transfer-arc
capacities. It is read straight off the base demands, so it does not depend on
any constructed or optimised solution.

The *load-dependent fuel* of a route is the pollution-routing form,

    sum_over_arcs theta * d_ij * [rho0 * x_ij + (rhoQ - rho0)/Q * f_ij]

which for one vehicle on a fixed sequence collapses to
theta * distance * (rho0 + (rhoQ - rho0) * load_fraction) per arc. It is a
cost component, not the allocated resource: the paper prices it at c^f and
charges it in the objective, while the depot account is driven by demand
assigned (tex, eq:c20).

THETA IS NOT A FREE PARAMETER. The paper carries a single constant, the unit
fuel cost c^f = 0.10 (build_data_release.EMISSION_COST), and folds the old
fuel-to-emission conversion into it. The code still threads theta through
route_metrics and the bounding routines, so the two agree only while theta is
exactly 1. The assertion below enforces that, because a silent divergence here
would make every reported cost disagree with the formulation in print.
"""
from __future__ import annotations

THETA = 1.0
RHO_EMPTY = 0.9
RHO_FULL = 1.5

assert THETA == 1.0, (
    "The paper has no theta: it carries one unit fuel cost c^f and folds the "
    "conversion into it. Change EMISSION_COST in build_data_release.py "
    "instead, or remove theta from route_metrics, diagonal_bound and "
    "exact_model first."
)

# tex, subsec:experimental_setup: "lower and upper positioning bounds
# equal to 45% and 160% of that allocation."
BUDGET_LOWER_FRAC = 0.45
BUDGET_UPPER_FRAC = 1.60


def reference_demand_per_depot(depots, customers) -> dict:
    """Expected demand of each depot's home customers, which is the natural
    scale for the stock it should hold. Unlike a routing construction this
    does not depend on any algorithm, and allocating the expected total
    leaves roughly half the scenarios short and half long, so the
    rebalancing mechanism is active without further calibration."""
    ref: dict[int, float] = {d.idx: 0.0 for d in depots}
    for c in customers:
        ref[c.home_depot] = ref.get(c.home_depot, 0.0) + c.base_demand
    return ref


def corporate_stock(ref_by_depot: dict, gamma_B: float):
    """Total stock B^corp = gamma_B * (expected total demand), split across
    depots in proportion to their expected home demand, with positioning
    bounds at 45% and 160% of that split. gamma_B is a service-cover factor:
    below one the network is short in expectation, above one it is long."""
    ref_total = sum(ref_by_depot.values())
    if ref_total <= 0:
        raise ValueError("total reference demand must be strictly positive")
    B_corp = gamma_B * ref_total
    alloc = {d: B_corp * r / ref_total for d, r in ref_by_depot.items()}
    bounds = {d: (BUDGET_LOWER_FRAC * a, BUDGET_UPPER_FRAC * a) for d, a in alloc.items()}
    return B_corp, alloc, bounds

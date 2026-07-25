"""Base demand and scenario generation.

Implements the "Demand streams" paragraph of Appendix "Instance and scenario
generation" and equation (scenario_generator) in
t-mdvrp_CBAR_PPBRC_public_benchmark_protocol.tex:

    q_i^omega = qbar_i * exp(sigma Z_{h(i)}^omega - sigma^2/2)
                       * exp(sigma_eps eps_i^omega - sigma_eps^2/2)

Z^omega is a depot-level Gaussian factor with equicorrelation rho;
eps_i^omega is an independent customer shock. The truncation/resampling rule
implemented here matches the tex literally: redraw only to maintain
positivity (automatic, since a lognormal multiplier is always positive) and
individual-vehicle feasibility (q_i^omega <= vehicle capacity). No aggregate
fleet-feasibility check is added on top of that; geometry.TARGET_UTILIZATION
is instead kept well under 1 so aggregate infeasibility is negligible by
construction, per the paper's own assumption in the Notation section that
"every scenario is operationally feasible".
"""
from __future__ import annotations

import math

import numpy as np

STUDENT_T_DF = 5
_STUDENT_T_STD = math.sqrt(STUDENT_T_DF / (STUDENT_T_DF - 2))  # unit-variance rescale


def equicorrelation_sqrt(n: int, rho: float) -> np.ndarray:
    """Symmetric square root of an n x n equicorrelation matrix.

    Uses an eigendecomposition rather than a Cholesky factor because E4's
    negative-correlation arm (rho = -0.25 at n = 5 depots) sits exactly on
    the equicorrelation PSD boundary rho = -1/(n-1); a Cholesky factor is
    numerically fragile there while eigh with eigenvalue clipping is not.
    """
    if n <= 1:
        return np.zeros((max(n, 1), max(n, 1)))
    lower = -1.0 / (n - 1)
    if rho < lower - 1e-9:
        raise ValueError(f"rho={rho} violates the equicorrelation PSD bound {lower} for n={n}")
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    eigval, eigvec = np.linalg.eigh(corr)
    eigval = np.clip(eigval, 0.0, None)
    return eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T


def base_demands(customers, depots, base_cv: float, depot_scale_cv: float,
                  rng: np.random.Generator, target_mean: float) -> None:
    """tex: "Base demands are positive lognormal draws with depot-specific
    scales" and (Geometry paragraph) "Depot demand scales are heterogeneous
    and are generated independently of the scenario streams." Each depot
    first draws its own mean-corrected lognormal scale factor around
    target_mean; its customers then draw base demand lognormally around
    that depot-specific mean. Mutates customers in place. Callers should
    pass the geometry rng stream, not a scenario stream, per the tex quote
    above.
    """
    by_depot: dict[int, list] = {}
    for c in customers:
        by_depot.setdefault(c.home_depot, []).append(c)
    sigma = math.sqrt(math.log(1 + base_cv ** 2))
    depot_scale_sigma = math.sqrt(math.log(1 + depot_scale_cv ** 2))
    for d in depots:
        members = by_depot.get(d.idx, [])
        if not members:
            continue
        scale = float(np.exp(rng.normal(-0.5 * depot_scale_sigma ** 2, depot_scale_sigma)))
        mu = math.log(target_mean * scale) - 0.5 * sigma ** 2
        for c in members:
            c.base_demand = float(np.exp(rng.normal(mu, sigma)))


def scenario_generator(customers, n_depots: int, n_scenarios: int, seed: int,
                        sigma_common: float, sigma_idio: float, rho: float,
                        vehicle_capacity: float, student_t: bool = False,
                        regime_prob: float = 0.0, regime_multiplier: float = 1.5) -> list[dict]:
    rng = np.random.default_rng(seed)
    sqrt_corr = equicorrelation_sqrt(n_depots, rho)
    scenarios = []
    for s in range(n_scenarios):
        for _attempt in range(1000):
            if student_t:
                raw = rng.standard_t(STUDENT_T_DF, size=n_depots) / _STUDENT_T_STD
            else:
                raw = rng.standard_normal(n_depots)
            z = sqrt_corr @ raw
            if student_t and regime_prob > 0 and rng.random() < regime_prob:
                # ASSUMPTION: "low-probability asymmetric depot-demand
                # regime" is not spelled out numerically in the tex. One
                # randomly chosen depot receives an extra multiplicative
                # surge (regime_multiplier) on top of its Gaussian/t factor.
                z[rng.integers(n_depots)] += math.log(regime_multiplier) / max(sigma_common, 1e-9)
            eps = rng.standard_normal(len(customers))
            q: dict[int, float] = {}
            ok = True
            for i, c in enumerate(customers):
                mult = math.exp(sigma_common * z[c.home_depot] - 0.5 * sigma_common ** 2)
                mult *= math.exp(sigma_idio * eps[i] - 0.5 * sigma_idio ** 2)
                val = c.base_demand * mult
                if val > vehicle_capacity:
                    ok = False
                    break
                q[c.idx] = val
            if ok:
                scenarios.append({"scenario": s + 1, "probability": 1.0 / n_scenarios, "demand": q})
                break
        else:
            raise RuntimeError("failed to sample a feasible scenario after 1000 attempts")
    return scenarios

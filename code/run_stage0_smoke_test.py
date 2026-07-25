#!/usr/bin/env python3
"""Stage 0 smoke test: rebuilt exact_model.py / recourse.py / budget_master.py
against the newly generated data, mirroring the earlier Phase A checks
(memory cbar_mdvrp_status.md / plans/logical-splashing-pixel.md):

  1. Exact MILP solves the smallest E1 instance: feasible/bounded.
  2. Recourse LP primal/dual strong duality holds to <1e-6, on both a random
     g and the g implied by the exact model's own emission solution.
  3. Complementary price conditions hold (Prop. complementarity).
  4. Institutional nesting Z_pool <= Z_network <= Z_independent holds for
     the same g (Cor. institutional_nesting).
  5. VSS >= 0 using the exact model's per-scenario emissions.
"""
from __future__ import annotations

import random
import sys

from instance_io import load_instance
import exact_model
import recourse
import budget_master

TOL = 1e-6


def close_enough(a: float, b: float, rel: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"D:\CBAR_MDVRP\data\CBAR_MDVRP_DATA_v1\E1\e1_c10_d2_w3_g1.json"
    inst = load_instance(path)
    print(f"loaded {inst.suite}/{inst.name}: C={inst.n_customers} D={inst.n_depots} "
          f"vehicles={len(inst.vehicles)}")

    scenarios = inst.scenarios("train")
    print(f"\n[1] exact MILP on {len(scenarios)} scenarios...")
    result = exact_model.build_and_solve(inst, scenarios, time_limit=120.0)
    print(f"    status={result['status']}  objective={result['objective']}  "
          f"runtime={result['runtime']:.1f}s")
    ok1 = result["objective"] is not None
    print("    PASS" if ok1 else "    FAIL")

    D = [d["idx"] for d in inst.depots]

    print("\n[2]+[3] recourse primal/dual duality + complementarity, random g...")
    rng = random.Random(0)
    g = {d: rng.uniform(-20, 20) for d in D}
    rec = recourse.solve_recourse(g, D, inst.transfer_arcs, inst.prices)
    gap = rec["duality_gap"]
    print(f"    primal={rec['primal']['value']:.6f}  dual={rec['dual']['value']:.6f}  gap={gap:.2e}")
    comp_errors = recourse.check_complementarity(rec["primal"], rec["dual"], inst.transfer_arcs,
                                                  inst.prices["buy"], inst.prices["sell"])
    ok23 = close_enough(rec["primal"]["value"], rec["dual"]["value"]) and not comp_errors
    print(f"    complementarity errors: {comp_errors or 'none'}")
    print("    PASS" if ok23 else "    FAIL")

    print("\n[4] institutional nesting Z_pool <= Z_network <= Z_independent...")
    # Deliberately construct a real deficit/surplus split (one depot short,
    # the rest long) so the transfer mechanism is actually exercised --
    # a random draw can trivially make every depot the same sign, in which
    # case all three network variants coincide and the check is uninformative.
    g2 = {d: -20.0 for d in D}
    g2[D[0]] = 20.0 * (len(D) - 1)
    independent_arcs: list[dict] = []
    pooled_arcs = [{"from": d1, "to": d2, "friction": 0.0, "capacity": 1e9}
                   for d1 in D for d2 in D if d1 != d2]
    z_independent = recourse.solve_recourse(g2, D, independent_arcs, inst.prices)["value"]
    z_network = recourse.solve_recourse(g2, D, inst.transfer_arcs, inst.prices)["value"]
    z_pooled = recourse.solve_recourse(g2, D, pooled_arcs, inst.prices)["value"]
    print(f"    Z_pool={z_pooled:.6f}  Z_network={z_network:.6f}  Z_independent={z_independent:.6f}")
    ok4 = z_pooled <= z_network + TOL and z_network <= z_independent + TOL
    print("    PASS" if ok4 else "    FAIL")

    print("\n[5] VSS >= 0 using the exact model's per-scenario emissions...")
    E_by_scenario = {}
    for w in range(len(scenarios)):
        E_by_scenario[w] = {d: result["vars"]["E"][d, w].value() for d in D}
    pi = {w: scenarios[w]["probability"] for w in range(len(scenarios))}
    vss_result = budget_master.value_of_stochastic_solution(
        E_by_scenario, pi, D, inst.transfer_arcs, inst.prices,
        inst.depot_budget_bounds, inst.corporate_budget)
    print(f"    stochastic={vss_result['stochastic']:.6f}  "
          f"wait_and_see={vss_result['wait_and_see']:.6f}  VSS={vss_result['vss']:.6f}")
    ok5 = vss_result["vss"] >= -TOL
    print("    PASS" if ok5 else "    FAIL")

    all_ok = ok1 and ok23 and ok4 and ok5
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

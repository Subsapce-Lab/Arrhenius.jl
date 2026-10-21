#!/usr/bin/env python3
"""Cantera reference counts for the prescribed-temperature burner card case.

The card case is solved the way Cantera solves a prescribed-temperature burner:
the energy equation is disabled and the direct (auto=False) solve is used, with
the same refine criteria as the native solve. The printed counts are the
reference that the native regression test asserts against.

Usage: fixed_t_card_counts.py spec.json [output.npz]
"""
import json
import os
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import cantera as ct
import numpy as np


def main(spec_path, out_path):
    spec = json.load(open(spec_path, encoding="utf-8"))
    gas = ct.Solution(spec["mechanism"] + ".yaml")
    gas.TPX = spec["T"], spec["P_atm"] * ct.one_atm, spec["X"]
    f = ct.BurnerFlame(gas, width=spec["width"])
    f.burner.mdot = spec["mdot"]
    f.transport_model = spec["transport"]
    f.set_refine_criteria(ratio=spec["ratio"], slope=spec["slope"], curve=spec["curve"],
                          prune=min(spec.get("prune", 0.0), spec["slope"], spec["curve"]))
    f.set_initial_guess()
    z = np.asarray(spec["profile_z"], float)
    T = np.asarray(spec["profile_T"], float)
    f.flame.set_fixed_temp_profile(z / z[-1], T)
    f.energy_enabled = False
    f.solve(loglevel=0, auto=False)
    st = f.solver_stats
    jac = st.get("jacobian_evals", [])
    steps = st.get("steps", [])
    grids = st.get("grid_points", [])
    print("CT_CARD id={} points={} jacobian_evals={} jac_sum={} steps={} grids={} "
          "energy_enabled={}".format(
              spec["id"], len(f.grid), jac, sum(jac), steps, grids, f.energy_enabled))
    print("   Tmax={} Ysum={} mdot={}".format(
        float(np.max(f.T)), float(np.max(np.abs(f.Y.sum(axis=0) - 1))), f.burner.mdot))
    # Cantera grid_points statistic sums ALL domains (OneDim::gridSize() = m_pts):
    # the flame domain plus the burner Inlet1D and the outlet Outlet1D, each
    # contributing one point.
    increments = [grids[i + 1] - grids[i] for i in range(len(grids) - 1)]
    print("   reported_increments={} flame_domain_points={}".format(
        increments, len(f.grid)))
    if out_path:
        np.savez_compressed(out_path, grid=np.asarray(f.grid), T=np.asarray(f.T),
            Y=np.asarray(f.Y), velocity=np.asarray(f.velocity))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit("usage: fixed_t_card_counts.py spec.json [output.npz]")
    main(args[0], args[1] if len(args) > 1 else None)

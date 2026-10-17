"""Print fixed-flame Cantera counts with production-adapter settings.

Usage: ct_fixed_counts.py SPEC [auto=true|false] [prune=0.0]
"""
import json
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cantera as ct
import numpy as np

spec_path = sys.argv[1]
overrides = dict(arg.split("=", 1) for arg in sys.argv[2:])
spec = json.load(open(spec_path))
gas = ct.Solution(spec["mechanism"] + ".yaml")
gas.TPX = spec["T"], spec["P_atm"] * ct.one_atm, spec["X"]
flame = ct.BurnerFlame(gas, width=spec["width"])
flame.burner.mdot = spec["mdot"]
flame.flame.set_fixed_temp_profile(
    np.asarray(spec["profile_z"]) / spec["width"], np.asarray(spec["profile_T"]))
flame.energy_enabled = False
flame.transport_model = spec["transport"]
prune = float(overrides.get("prune", min(spec.get("prune", 0.0), spec["slope"], spec["curve"])))
auto = overrides.get("auto", "false").lower() == "true"
criteria = (spec["ratio"], spec["slope"], spec["curve"], prune)
flame.set_refine_criteria(ratio=criteria[0], slope=criteria[1], curve=criteria[2], prune=criteria[3])
flame.solve(loglevel=0, auto=auto)
stats = flame.solver_stats
print("CTFIXED", json.dumps({
    "id": spec["id"], "auto": auto, "ratio_slope_curve_prune": criteria,
    "prune_source": "production=min(spec.prune or 0, slope, curve)",
    "initial_guess": "fresh BurnerFlame constructor; fixed profile applied; no restart",
    "transport": spec["transport"], "energy_enabled": flame.energy_enabled,
    "profile_z_normalized_by": spec["width"], "max_grid_points": "Cantera default",
    "jacobians": int(sum(stats.get("jacobian_evals", []))),
    "linear_solves": int(sum(stats.get("linear_solves", []))),
    "accepted_steps": int(sum(stats.get("steps", []))),
    "regrids": max(0, len(stats.get("grid_points", [])) - 1),
    "points": len(flame.grid), "converged": True
}, sort_keys=True))

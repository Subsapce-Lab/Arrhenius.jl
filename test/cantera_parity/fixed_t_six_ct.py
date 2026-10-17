#!/usr/bin/env python3
import json
import os
import sys
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import cantera as ct
import numpy as np

spec = json.load(open(sys.argv[1], encoding="utf-8"))
out = sys.argv[2]
gas = ct.Solution(spec["mechanism"] + ".yaml")
gas.TPX = spec["T"], spec["P_atm"] * ct.one_atm, spec["X"]
inlet_y = gas.Y.copy()
f = ct.BurnerFlame(gas, width=spec["width"])
f.burner.mdot = spec["mdot"]
f.transport_model = spec["transport"]
f.set_refine_criteria(ratio=spec["ratio"], slope=spec["slope"], curve=spec["curve"],
                       prune=min(spec.get("prune", 0.0), spec["slope"], spec["curve"]))
if spec["kind"] == "fixed":
    f.set_initial_guess()
    z, T = np.asarray(spec["profile_z"], float), np.asarray(spec["profile_T"], float)
    f.flame.set_fixed_temp_profile(z / z[-1], T)
    f.energy_enabled = False
elif spec["kind"] != "burner":
    raise ValueError("unsupported case kind " + spec["kind"])
f.solve(loglevel=0, auto=(spec["kind"] == "burner"))
s = f.solver_stats
names = "\n".join(gas.species_names).encode("utf-8")
np.savez_compressed(out, grid=np.asarray(f.grid), T=np.asarray(f.T), Y=np.asarray(f.Y),
    inlet_Y=inlet_y, species_names_utf8=np.frombuffer(names, dtype=np.uint8),
    jacobians=np.array([sum(s.get("jacobian_evals", []))]),
    time_steps=np.array([sum(s.get("steps", []))]),
    regrids=np.array([max(0, len(s.get("grid_points", [])) - 1)]))
print("CT case={} kind={} points={} jacobians={} newton={} steps={} regrids={}".format(
    spec["id"], spec["kind"], len(f.grid), sum(s.get("jacobian_evals", [])),
    sum(s.get("steps", [])), max(0, len(s.get("grid_points", [])) - 1)))

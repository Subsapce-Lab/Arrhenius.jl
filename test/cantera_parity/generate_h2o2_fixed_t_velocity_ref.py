#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import cantera as ct

if len(sys.argv) != 2:
    raise SystemExit("usage: generate_h2o2_fixed_t_velocity_ref.py output.npz")
root = Path(__file__).resolve().parents[2]
mechanism = root / "mechanism" / "h2o2.yaml"
grid = 0.03 * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
gas = ct.Solution(str(mechanism))
gas.TPX = 300.0, ct.one_atm, "H2:1.1,O2:1,AR:5"
flame = ct.BurnerFlame(gas, grid=grid)
flame.burner.T = 300.0
flame.burner.X = gas.X
flame.burner.mdot = 0.07
flame.flame.set_fixed_temp_profile([0.0, 0.25, 1.0], [300.0, 1600.0, 300.0])
flame.energy_enabled = False
flame.set_initial_guess()
flame.solve(loglevel=0, refine_grid=False, auto=False)
np.savez_compressed(sys.argv[1], grid=flame.grid, T=flame.T,
    velocity=flame.velocity, Y=flame.Y)
print("CT_FIXED_VELOCITY points={} path={}".format(len(flame.grid), sys.argv[1]))

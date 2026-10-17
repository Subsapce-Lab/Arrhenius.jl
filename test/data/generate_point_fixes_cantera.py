#!/usr/bin/env python3
"""Generate point_fixes_cantera.csv with the configured Cantera build."""
import csv
from pathlib import Path
import cantera as ct

root = Path(__file__).resolve().parents[2]
gas = ct.Solution(str(root / "mechanism" / "h2o2.yaml"), transport_model="mixture-averaged")
rows = [("cantera_version", "build", ct.__version__)]
gas.TPX = 1400.0, ct.one_atm, "H2:2,O2:1,AR:4"
base = gas.Y.copy()
rho0 = gas.density
h, ar = gas.species_index("H"), gas.species_index("AR")
for y in (0.0, 1e-25, -1e-25, -1e-7):
    Y = base.copy()
    Y[h] = y
    Y[ar] += base[h] - y
    gas.set_unnormalized_mass_fractions(Y)
    gas.TP = 1400.0, ct.one_atm
    rows.append(("reactor_C_H", format(y, ".17g"), format(gas.concentrations[h], ".17g")))
    gas.set_unnormalized_mass_fractions(Y)
    gas.TD = 1400.0, rho0
    rows.append(("network_C_H", format(y, ".17g"), format(gas.concentrations[h], ".17g")))
plog_rates = [(pressure, ct.ArrheniusRate(1.0, 0.0, 1e9))
              for pressure in (ct.one_atm, ct.one_atm, 10*ct.one_atm, 10*ct.one_atm)]
plog = ct.PlogRate(plog_rates)
for pressure in (0.1*ct.one_atm, 3*ct.one_atm, 100*ct.one_atm):
    rows.append(("plog_floor", format(pressure/ct.one_atm, ".17g"),
                 format(plog(1.0, pressure), ".17g")))
for label, X in (("pure", {"H2": 1.0}), ("near", {"H2": 1.0-1e-12, "O2": 1e-12})):
    gas.TPX = 1000.0, ct.one_atm, X
    idx = gas.species_index("H2")
    # Cantera Flow1D molar-gradient flames use getMixDiffCoeffs (Python mix_diff_coeffs).
    rows.append(("transport_mix_H2", label, format(gas.mix_diff_coeffs[idx], ".17g")))
    rows.append(("transport_mass_H2", label, format(gas.mix_diff_coeffs_mass[idx], ".17g")))
with (Path(__file__).parent / "point_fixes_cantera.csv").open("w", newline="") as stream:
    csv.writer(stream, lineterminator="\n").writerows(rows)

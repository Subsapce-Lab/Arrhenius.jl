#!/usr/bin/env python3
"""Check composition closure, premixed inlet-to-outlet elemental balance, and planar mass flux in solver archives."""
import argparse
import json
from pathlib import Path

import cantera as ct
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("manifest", type=Path, help="nh3_set_manifest.csv")
    parser.add_argument("spec_root", type=Path, help="nh3-parity checkout")
    parser.add_argument("mechanism_root", type=Path, help="Arrhenius mechanism directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    specs = {}
    for row in __import__("csv").DictReader(args.manifest.open(newline="")):
        specs[row["id"]] = row["spec"]
    results = []
    for archive_path in sorted(args.archive_dir.glob("*.npz")):
        spec_path = args.spec_root / specs[archive_path.stem]
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        mech_path = args.mechanism_root / (spec["mechanism"] + ".yaml")
        gas = ct.Solution(str(mech_path))
        with np.load(archive_path, allow_pickle=False) as data:
            y = np.asarray(data["Y"], dtype=np.float64)
            names = bytes(np.asarray(data["species_names_utf8"], dtype=np.uint8)).decode("utf-8").splitlines()
            if y.shape[0] != gas.n_species:
                y = y.T
            if names != list(gas.species_names):
                raise ValueError(f"species order mismatch in {archive_path.name}")
            if y.shape[0] != gas.n_species:
                raise ValueError(f"species axis mismatch in {archive_path.name}: {y.shape}")
            sums = y.sum(axis=0)
            sum_y_error = float(np.max(np.abs(sums - 1.0)))
            element_error = None
            if spec["kind"] in ("free", "burner", "fixed"):
                max_error = 0.0
                mw = np.asarray(gas.molecular_weights)
                aw = np.asarray(gas.atomic_weights)
                for e in range(gas.n_elements):
                    factors = np.asarray([gas.n_atoms(k, e) * aw[e] / mw[k] for k in range(gas.n_species)])
                    profile = factors @ y
                    max_error = max(max_error, float(abs(profile[-1] - profile[0])))
                element_error = max_error
            mass_flux_error = None
            if spec["kind"] in ("free", "burner", "fixed"):
                temperature = np.asarray(data["T"], dtype=np.float64)
                velocity = np.asarray(data["velocity"], dtype=np.float64)
                pressure = float(spec["P_atm"]) * ct.one_atm
                flux = np.empty(y.shape[1], dtype=np.float64)
                for j in range(y.shape[1]):
                    gas.TPY = temperature[j], pressure, y[:, j]
                    flux[j] = gas.density * velocity[j]
                scale = max(float(np.mean(np.abs(flux))), np.finfo(float).tiny)
                mass_flux_error = float(np.max(np.abs(flux - flux[0])) / scale)
        results.append({"id": archive_path.stem, "sumY_max": sum_y_error,
                        "element_inlet_outlet_error": element_error, "mass_flux_relative_spread": mass_flux_error})
    summary = {
        "case_count": len(results),
        "sumY_max": max((r["sumY_max"] for r in results), default=None),
        "element_inlet_outlet_max": max((r["element_inlet_outlet_error"] for r in results if r["element_inlet_outlet_error"] is not None), default=None),
        "mass_flux_relative_spread_max": max((r["mass_flux_relative_spread"] for r in results if r["mass_flux_relative_spread"] is not None), default=None),
        "cases": results,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Generate a Cantera adaptive criterion-path point for the production gate.

Runs the shared Cantera adapter once at a uniformly scaled set of refinement
criteria, then records the attained steady tolerance and same-state residual
norm required by the gate. Output belongs in a task-local SPREAD_DIR.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("scale", type=float, choices=(0.97, 1.03))
    parser.add_argument("--adapter", type=Path, required=True,
                        help="shared nh3_cantera_tight.py path")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    for key in ("ratio", "slope", "curve"):
        spec[key] = float(spec[key]) * args.scale
    spec["criterion_scale"] = args.scale
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cantera-spread-") as scratch:
        scaled_spec = Path(scratch) / "scaled-spec.json"
        scaled_spec.write_text(json.dumps(spec), encoding="utf-8")
        os.environ["TIGHT_RTOL"] = "1e-8"
        os.environ["TIGHT_ATOL"] = "1e-14"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        sys.argv = [str(args.adapter), str(scaled_spec), str(args.output),
                    "--single", "--batches=1", "--per-batch=1"]
        namespace = runpy.run_path(str(args.adapter), run_name="__main__")
    reached = namespace.get("TIGHT_REACHED")
    if reached is None or not np.isfinite(float(reached)) or float(reached) > 1e-8:
        raise SystemExit(f"Cantera adaptive solve did not reach rtol=1e-8: {reached!r}")
    flame = namespace["f"]
    flame.eval(rdt=0.0)
    residuals = []
    for domain in flame.domains:
        for component in range(domain.n_components):
            residuals.append(np.asarray(domain.residuals(domain.component_name(component)), dtype=float).ravel())
    if not residuals or not all(np.isfinite(values).all() for values in residuals):
        raise SystemExit("Cantera adaptive residual contains no finite component records")
    residual_inf = max(float(np.max(np.abs(values))) for values in residuals if values.size)
    if not np.isfinite(residual_inf):
        raise SystemExit(f"non-finite Cantera residual_inf: {residual_inf}")
    with np.load(args.output, allow_pickle=False) as archive:
        record = {key: np.asarray(archive[key]).copy() for key in archive.files}
    record["criterion_scale"] = np.asarray([args.scale])
    record["tight_rtol_reached"] = np.asarray([float(reached)])
    record["residual_inf"] = np.asarray([residual_inf])
    np.savez(args.output, **record)
    print(f"spread scale={args.scale:.2f} rtol={float(reached):g} residual_inf={residual_inf:.6g} points={len(record['grid'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

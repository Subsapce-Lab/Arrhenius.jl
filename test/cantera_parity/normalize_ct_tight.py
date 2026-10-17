#!/usr/bin/env python3
"""Normalize a Cantera tight reference archive for the current production gate.

The Cantera producer records tight_rtol_reached after each successful polish.
This adapter accepts only a final archive that reached 1e-8 and maps its finest
level to the gate's single-state schema; it does not alter the flame solution.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import tempfile

import numpy as np

FIELDS = (
    "grid", "T", "Y", "SL", "velocity", "mdot", "stats_jacobians",
    "stats_newton_iterations", "stats_time_steps", "stats_regrids",
    "stats_final_points",
)
EXPECTED_RTOL = 1.0e-8
EXPECTED_ATOL = 1.0e-14


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="nh3_cantera_tight.py --reference output")
    parser.add_argument("output", type=Path, help="task-local <case>.ct.npz output")
    args = parser.parse_args()
    with np.load(args.raw, allow_pickle=False) as archive:
        reached = np.asarray(archive["tight_rtol_reached"], dtype=float)
        if reached.size != 1 or not np.isfinite(reached).all() or reached.item() != EXPECTED_RTOL:
            raise SystemExit(f"final Cantera tight solve did not reach rtol={EXPECTED_RTOL:g}: {reached}")
        levels = sorted({int(key[1:].split("_", 1)[0]) for key in archive.files
                         if key.startswith("L") and "_" in key})
        if not levels:
            raise SystemExit("raw archive has no refined solution levels")
        level = levels[-1]
        record = {key: np.asarray(archive[f"L{level}_{key}"]).copy() for key in FIELDS}
        record["species_names"] = np.asarray(archive["species_names"]).copy()
        record["newton_ok"] = np.asarray([1.0])
        record["solver_rtol"] = np.asarray([EXPECTED_RTOL])
        record["solver_atol"] = np.asarray([EXPECTED_ATOL])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=args.output.name + ".", suffix=".npz", dir=args.output.parent)
    os.close(fd)
    try:
        np.savez(temporary_name, **record)
        Path(temporary_name).replace(args.output)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    print(f"normalized level L{level}; Cantera polish rtol={EXPECTED_RTOL:g}, atol={EXPECTED_ATOL:g}; {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

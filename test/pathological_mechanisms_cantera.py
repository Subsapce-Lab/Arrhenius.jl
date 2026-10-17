#!/usr/bin/env python3
"""Cantera reference behavior for generated pathological h2o2 fixtures.

Run in the pinned Cantera environment:
  python test/pathological_mechanisms_cantera.py --case clean
  python test/pathological_mechanisms_cantera.py --solves --case nasa7_small
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from pathlib import Path

import cantera as ct
import numpy as np
import yaml


def reaction(root, equation):
    return next(r for r in root["reactions"] if r.get("equation") == equation)


def fixtures(base, out):
    clean = yaml.safe_load(base.read_text(encoding="utf-8"))
    result = {"clean": copy.deepcopy(clean)}
    for name, delta in (("nasa7_small", 1e-3), ("nasa7_large", 1.0)):
        root = copy.deepcopy(clean)
        h2 = next(s for s in root["species"] if s["name"] == "H2")
        h2["thermo"]["data"][1][0] += delta
        result[name] = root
    root = copy.deepcopy(clean)
    r = reaction(root, "O + H2 <=> H + OH")
    rc = r.pop("rate-constant")
    r["type"] = "pressure-dependent-Arrhenius"
    r["rate-constants"] = [
        {"P": "0.1 atm", "A": rc["A"], "b": rc["b"], "Ea": 1e6},
        {"P": "10 atm", "A": rc["A"], "b": rc["b"], "Ea": 1e6},
    ]
    result["plog_underflow"] = root
    root = copy.deepcopy(clean)
    r = copy.deepcopy(reaction(root, "O + H2 <=> H + OH"))
    r["id"] = "duplicate-candidate"
    root["reactions"].append(r)
    result["duplicate_unflagged"] = root
    root = copy.deepcopy(clean)
    reaction(root, "O + H2 <=> H + OH")["equation"] = "O + H2 <=> H + H"
    result["unbalanced"] = root
    root = copy.deepcopy(clean)
    reaction(root, "O + H2 <=> H + OH")["rate-constant"]["A"] *= -1
    result["negative_A"] = root
    root = copy.deepcopy(clean)
    reaction(root, "O + H2 <=> H + OH")["rate-constant"]["Ea"] = 1e7
    result["extreme_Ea"] = root
    root = copy.deepcopy(clean)
    reaction(root, "O + H2 <=> H + OH")["rate-constant"]["A"] = 1e15
    result["collision_limit"] = root
    paths = {}
    for name, root in result.items():
        path = out / f"{name}.yaml"
        path.write_text(yaml.safe_dump(root, sort_keys=False), encoding="utf-8")
        paths[name] = path
    return paths


def evaluate(path, solves):
    row = {"path": path.name, "load": "ok", "evaluation": "not-run"}
    try:
        gas = ct.Solution(str(path))
    except Exception as exc:
        row.update(load="error", message=str(exc).splitlines()[0])
        return row
    temps = (250.0, 999.999, 1000.0, 1000.001, 2500.0, 3500.0)
    finite = True
    for T in temps:
        gas.TPX = T, ct.one_atm, "H2:2, O2:1, AR:4"
        finite &= all(np.all(np.isfinite(x)) for x in (
            gas.net_production_rates, gas.standard_cp_R, gas.standard_enthalpies_RT, gas.standard_entropies_R))
    row.update(evaluation="finite" if finite else "nonfinite", temperatures_K=temps)
    if not solves:
        return row
    try:
        flame_gas = ct.Solution(str(path))
        flame_gas.TP = 300.0, ct.one_atm
        flame_gas.set_equivalence_ratio(1.1, "H2", "O2:1, AR:5")
        flame = ct.FreeFlame(flame_gas, width=0.03)
        flame.set_refine_criteria(ratio=3.0, slope=0.06, curve=0.12)
        flame.solve(loglevel=0, auto=True)
        row["free_flame"] = {
            "converged": True, "grid_points": len(flame.grid),
            "speed_m_s": float(flame.velocity[0]), "solver_stats": flame.solver_stats}
    except Exception as exc:
        row["free_flame"] = {"converged": False, "message": str(exc).splitlines()[0]}
    try:
        ignition_gas = ct.Solution(str(path))
        ignition_gas.TPX = 1001.0, 3 * ct.one_atm, "H2:2, O2:1, AR:4"
        reactor = ct.IdealGasConstPressureReactor(ignition_gas, energy="on")
        network = ct.ReactorNet([reactor])
        t_end = 0.02
        while network.time < t_end and reactor.T < 1600:
            network.step()
        row["ignition"] = {
            "converged": math.isfinite(reactor.T), "time_s": float(network.time),
            "temperature_K": float(reactor.T),
            "steps": int(network.solver_stats.get("steps", 0))}
    except Exception as exc:
        row["ignition"] = {"converged": False, "message": str(exc).splitlines()[0]}
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solves", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1] / "mechanism" / "h2o2.yaml"
    with tempfile.TemporaryDirectory(prefix="pathomech-") as tmp:
        out = args.output_dir or Path(tmp)
        out.mkdir(parents=True, exist_ok=True)
        paths = fixtures(base, out)
        selected = [(n, p) for n, p in paths.items() if args.case is None or n == args.case]
        if args.case is not None and not selected:
            raise SystemExit(f"unknown case: {args.case}")
        rows = [evaluate(p, args.solves) for _, p in selected]
        print(json.dumps({"cantera_version": ct.__version__, "fixtures": rows}, indent=2))


if __name__ == "__main__":
    main()

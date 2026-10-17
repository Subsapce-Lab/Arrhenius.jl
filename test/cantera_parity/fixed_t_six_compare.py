#!/usr/bin/env python3
import json
import sys
import numpy as np
import cantera as ct

def compare_case(spec_path, native_path, cantera_path):
    spec = json.load(open(spec_path, encoding="utf-8"))
    arr = np.load(native_path, allow_pickle=False)
    ref = np.load(cantera_path, allow_pickle=False)
    gas = ct.Solution(spec["mechanism"] + ".yaml")
    names = [gas.element_name(m) for m in range(gas.n_elements)]
    comp = np.array([[gas.n_atoms(k, m) * gas.atomic_weights[m] / gas.molecular_weights[k]
                      for m in range(gas.n_elements)] for k in range(gas.n_species)])

    def unpack_names(d):
        return bytes(d["species_names_utf8"].tolist()).decode("utf-8").split("\n")

    def errors(d):
        order = [unpack_names(d).index(name) for name in gas.species_names]
        inlet = comp.T @ d["inlet_Y"][order]
        outlet = comp.T @ d["Y"][order, -1]
        return {name: abs(float(a - b)) for name, a, b in zip(names, inlet, outlet)}

    native_error, ct_error = errors(arr), errors(ref)
    allowed = {name: max(1.02 * ct_error[name], 1e-5) for name in names}
    inlet_temperature = max(float(spec.get("T", 300.0)), float(spec.get("T_surface", 0.0) or 0.0))
    burning_rise = float(np.max(ref["T"])) - inlet_temperature
    reference_burning = burning_rise >= 500.0
    counts = {}
    for key in ("jacobians", "time_steps", "regrids", "failures"):
        if key in arr.files and key in ref.files:
            counts[key] = {"arrhenius": int(arr[key][0]), "cantera": int(ref[key][0])}
    result = {"case": spec["id"], "reference_burning_rise_K": burning_rise,
              "reference_burning": reference_burning, "native_element_error": native_error,
              "cantera_element_error": ct_error, "gate_allowance": allowed,
              "element_pass": {name: native_error[name] <= allowed[name] for name in names},
              "counts": counts}
    print(json.dumps(result, sort_keys=True))


args = sys.argv[1:]
if len(args) < 3 or len(args) % 3:
    raise SystemExit("usage: fixed_t_six_compare.py spec.json native.npz cantera.npz [spec.json native.npz cantera.npz ...]")
for i in range(0, len(args), 3):
    compare_case(*args[i:i + 3])

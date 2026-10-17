#!/usr/bin/env python3
import json
import sys
import numpy as np
import cantera as ct

spec = json.load(open(sys.argv[1], encoding="utf-8"))
arr = np.load(sys.argv[2], allow_pickle=False)
ref = np.load(sys.argv[3], allow_pickle=False)
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
counts = {}
for key in ("jacobians", "time_steps", "regrids", "failures"):
    if key in arr.files and key in ref.files:
        counts[key] = {"arrhenius": int(arr[key][0]), "cantera": int(ref[key][0])}
result = {"case": spec["id"], "native_element_error": native_error,
          "cantera_element_error": ct_error, "gate_allowance": allowed,
          "element_pass": {name: native_error[name] <= allowed[name] for name in names},
          "counts": counts}
print(json.dumps(result, sort_keys=True))

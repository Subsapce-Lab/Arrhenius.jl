#!/usr/bin/env python3
"""Evaluate Cantera's fixed-grid Newton correction and conditioning at a stored state."""
import os, sys
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import cantera as ct
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, onenormest, splu
sys.path.insert(0, os.path.dirname(__file__))
import xresid
spec = xresid.load_spec(sys.argv[1])
if os.environ.get("MECHANISM_DIR") and not os.path.isabs(spec["mechanism"]):
    spec["mechanism"] = os.path.join(os.environ["MECHANISM_DIR"], spec["mechanism"])
species = ct.Solution(xresid.mechanism_path(spec)).species_names
can = xresid.canonical(sys.argv[2], species)
f = xresid.build(spec, can["grid"])
xresid.install(f, can)
f.eval(rdt=0.0)
components, values, residuals, tolerances = [], [], [], []
for domain in f.domains:
    for i in range(domain.n_components):
        name = domain.component_name(i)
        x = np.asarray(domain.values(name), float)
        r = np.asarray(domain.residuals(name), float)
        components.append((domain.name, name))
        values.append(x)
        residuals.append(r)
        tolerances.append(np.full(x.shape, 1e-8 * np.mean(np.abs(x)) + 1e-14))
# Cantera's OneDim ordering is point-major within each domain.
blocks_x = [np.column_stack([values[i] for i in range(k, k + d.n_components)])
            for d in f.domains if d.n_components for k in [sum(x.n_components for x in f.domains[:f.domains.index(d)])]]
blocks_r = [np.column_stack([residuals[i] for i in range(k, k + d.n_components)])
            for d in f.domains if d.n_components for k in [sum(x.n_components for x in f.domains[:f.domains.index(d)])]]
blocks_w = [np.column_stack([tolerances[i] for i in range(k, k + d.n_components)])
            for d in f.domains if d.n_components for k in [sum(x.n_components for x in f.domains[:f.domains.index(d)])]]
x = np.concatenate([b.ravel() for b in blocks_x])
r = np.concatenate([b.ravel() for b in blocks_r])
w = np.concatenate([b.ravel() for b in blocks_w])
f.linear_solver = ct.EigenSparseDirectJacobian()
f.eval_jacobian()
J = f.linear_solver.jacobian
if not sparse.issparse(J):
    J = sparse.csc_matrix(J)
else:
    J = J.tocsc()
if len(sys.argv) > 3:
    active_names = ["T", *species, "velocity"]
    component_names = [name for domain, name in components]
    active_components = [component_names.index(name) for name in active_names]
    ncomponents = len(component_names)
    active = np.asarray([j*ncomponents+k for j in range(len(can["grid"]))
                         for k in active_components], dtype=np.int64)
    Jactive = J[active, :][:, active].tocsc()
    np.savez_compressed(sys.argv[3], r_ct=r[active], weights=w[active],
                        dofs_full=np.asarray([r.size]), shape=np.asarray(Jactive.shape),
                        J_data=Jactive.data, J_indices=Jactive.indices, J_indptr=Jactive.indptr,
                        grid=can["grid"], T=can["T"], Y=can["Y"], velocity=can["velocity"],
                        molecular_weights=np.asarray(f.gas.molecular_weights),
                        species_names=np.asarray(species, dtype="U"))
lu = splu(J)
dx = lu.solve(-r)
step = float(np.linalg.norm(dx / w) / np.sqrt(len(dx)))
op = LinearOperator(J.shape, matvec=lambda v: J @ v, rmatvec=lambda v: J.T @ v, dtype=float)
iop = LinearOperator(J.shape, matvec=lambda v: lu.solve(v), rmatvec=lambda v: lu.solve(v, trans="T"), dtype=float)
cond1 = float(onenormest(op) * onenormest(iop))
print(f"case={spec['id']} points={len(can['grid'])} dofs={len(r)} sparse_nnz={J.nnz}")
print(f"residual_inf={np.linalg.norm(r, np.inf):.9g} weighted_step_rms={step:.9g} cond1_est={cond1:.9g}")
print("component_order=" + ",".join(f"{d}:{c}" for d,c in components))
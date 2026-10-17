# Solve both cross-code linear systems and the two diagonal systems in CT physical coordinates.
# Usage: python fixed_t_matrix_cross.py cantera.npz arrhenius.npz case_id
import sys, numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
ct=np.load(sys.argv[1]); ar=np.load(sys.argv[2]); name=sys.argv[3]
rct=ct["r_ct"]; rarr=ar["r_arr"]; weights=ct["weights"]
shape=tuple(int(x) for x in ct["shape"])
Jct=csc_matrix((ct["J_data"],ct["J_indices"],ct["J_indptr"]),shape=shape)
lu=splu(Jct)
bw=int(ar["bw"][0]); bands=ar["bands_arr"]
def norm(dx): return float(np.linalg.norm(dx/weights)/np.sqrt(dx.size))
dx_aa=solve_banded((bw,bw),bands,-rarr,check_finite=False)
dx_ca=solve_banded((bw,bw),bands,-rct,check_finite=False)
dx_cc=lu.solve(-rct)
dx_ac=lu.solve(-rarr)
print(f"case={name} Arr_resid/Arr_J={norm(dx_aa):.12g} CT_resid/Arr_J={norm(dx_ca):.12g} CT_resid/CT_J={norm(dx_cc):.12g} Arr_resid/CT_J={norm(dx_ac):.12g}")

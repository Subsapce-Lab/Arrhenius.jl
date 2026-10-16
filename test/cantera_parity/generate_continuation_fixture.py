"""Generate a fixed-grid Cantera two-point-control H2/O2 continuation fixture.

Requires Cantera 4.0.0a2 and writes a compressed NPZ fixture to argv[1]. The control update follows
samples/python/onedim/diffusion_flame_continuation.py at commit
726522be4e2a13454d8415b7ef799d621f665cf3, with refinement disabled so every
stored solution uses the same fixed grid.
"""
import json, sys
from pathlib import Path
import cantera as ct
import numpy as np

out=sys.argv[1]
mechanism=Path(__file__).resolve().parents[2]/"mechanism"/"h2o2.yaml"
npoints=int(sys.argv[2]) if len(sys.argv)>2 else 12
maxsteps=int(sys.argv[3]) if len(sys.argv)>3 else 250
f=ct.CounterflowDiffusionFlame(ct.Solution(str(mechanism)),width=.018)
f.P=1e5
f.fuel_inlet.mdot=.5; f.fuel_inlet.X="H2:1"; f.fuel_inlet.T=300
f.oxidizer_inlet.mdot=3.; f.oxidizer_inlet.X="O2:1"; f.oxidizer_inlet.T=500
f.flame.grid=np.linspace(0,.018,npoints)
f.solve(loglevel=0,refine_grid=False,auto=False)
f.two_point_control_enabled=True
f.flame.set_bounds(spread_rate=(-1e-5,1e20))
f.max_time_step_count=100
initial_max=max(f.T); amax=strain=f.strain_rate("max")
step=20.; rows=[]; errors=0
for i in range(maxsteps):
    spacing=.6 if strain>.98*amax else .95
    target=float(np.min(f.T)+spacing*(np.max(f.T)-np.min(f.T)))
    backup=f.to_array()
    f.set_left_control_point(target); f.set_right_control_point(target)
    f.left_control_point_temperature-=step; f.right_control_point_temperature-=step
    if f.left_control_point_temperature<f.fuel_inlet.T+100 or f.right_control_point_temperature<f.oxidizer_inlet.T+100:
        break
    try:
        f.solve(loglevel=0,refine_grid=False,auto=False)
        errors=0
    except ct.CanteraError:
        f.from_array(backup); step*=.7; errors+=1
        if errors>=3: break
        continue
    current=f.strain_rate("max")
    rows.append({"index":i,"Tmax_K":float(max(f.T)),"strain_rate_per_s":float(current),
        "grid_m":np.asarray(f.grid).tolist(),"T_K":np.asarray(f.T).tolist(),
        "Y":np.asarray(f.Y).T.tolist(),"velocity_m_s":np.asarray(f.velocity).tolist(),
        "fuel_mdot":float(f.fuel_inlet.mdot),"oxidizer_mdot":float(f.oxidizer_inlet.mdot),
        "spreadRate":np.asarray(f.flame.values("spreadRate")).tolist(),
        "Lambda":np.asarray(f.flame.values("Lambda")).tolist(),
        "solver_stats":f.solver_stats})
    if abs(max(f.T)-initial_max)<16: step=min(step+3.,100.)
    elif abs(max(f.T)-initial_max)>20: step*=18./abs(max(f.T)-initial_max)
    strain=current; amax=max(current,amax); initial_max=max(f.T)
    if current/amax<.05: break
turn=int(np.argmax([r["strain_rate_per_s"] for r in rows])) if rows else -1
turnrow=rows[turn] if turn>=0 else {}
np.savez_compressed(out,grid_points=np.asarray([npoints]),pressure_Pa=np.asarray([1e5]),
    fuel_mdot=np.asarray([.5]),oxidizer_mdot=np.asarray([3.]),
    grid_m=np.asarray(turnrow.get("grid_m",f.grid)),
    Tmax_curve_K=np.asarray([r["Tmax_K"] for r in rows]),
    strain_rate_curve_per_s=np.asarray([r["strain_rate_per_s"] for r in rows]),
    turning_index=np.asarray([turn]),turning_T_K=np.asarray(turnrow.get("T_K",f.T)),
    turning_Y=np.asarray(turnrow.get("Y",f.Y.T)),
    turning_velocity_m_s=np.asarray(turnrow.get("velocity_m_s",f.velocity)),
    turning_Tmax_K=np.asarray([turnrow.get("Tmax_K",max(f.T))]),
    turning_strain_rate_per_s=np.asarray([turnrow.get("strain_rate_per_s",f.strain_rate("max"))]),
    turning_fuel_mdot=np.asarray([turnrow.get("fuel_mdot",f.fuel_inlet.mdot)]),
    turning_oxidizer_mdot=np.asarray([turnrow.get("oxidizer_mdot",f.oxidizer_inlet.mdot)]),
    turning_spreadRate=np.asarray(turnrow.get("spreadRate",f.flame.values("spreadRate"))),
    turning_Lambda=np.asarray(turnrow.get("Lambda",f.flame.values("Lambda"))),
    grid=np.asarray(turnrow.get("grid_m",f.grid)),
    T=np.asarray(turnrow.get("T_K",f.T)),
    Y=np.asarray(turnrow.get("Y",f.Y.T)),
    velocity=np.asarray(turnrow.get("velocity_m_s",f.velocity)),
    spreadRate=np.asarray(turnrow.get("spreadRate",f.flame.values("spreadRate"))),
    Lambda=np.asarray(turnrow.get("Lambda",f.flame.values("Lambda"))))
print(json.dumps({"points":len(rows),"turning_index":turn,
    "turning_strain":float(turnrow.get("strain_rate_per_s",float("nan"))),
    "turning_Tmax":float(turnrow.get("Tmax_K",float("nan"))),
    "last_strain":rows[-1]["strain_rate_per_s"] if rows else None,"grid_points":len(f.grid)}))

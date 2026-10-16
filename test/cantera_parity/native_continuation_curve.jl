using Arrhenius, NPZ, Printf

np=parse(Int,get(ENV,"CONTINUATION_GRID_POINTS","266"))
steps=parse(Int,get(ENV,"CONTINUATION_STEPS","140"))
mechanism=joinpath(dirname(pathof(Arrhenius)),"..","mechanism","h2o2.yaml")
fixture=joinpath(@__DIR__,"..","data","cantera_h2o2_two_point_$(np)pt.npz")
ref=NPZ.npzread(fixture)
gas=CreateSolution(mechanism)
width=.018
flame=CounterflowDiffusionFlame(gas;fuel="H2:1",oxidizer="O2:1",
    mdot_fuel=.5,mdot_oxidizer=3.,T_fuel=300.,T_oxidizer=500.,P=1e5,width,
    grid=collect(range(0,width;length=np)))
solve!(flame;refine_grid=false,max_time_steps=200)
curve=pseudo_arclength!(flame;parameter=:inlet_flux_scale,steps,ds=.05,
    min_ds=.05/16,max_ds=.1,tolerance=1e-8)
peak=argmax([p.max_strain_rate for p in curve])
p=curve[peak]
@printf("grid=%d points=%d native peak strain=%.9g Tmax=%.9g residual=%.3g turning_flag=%s\n",
    np,length(curve),p.max_strain_rate,p.max_temperature,p.residual_inf,string(p.turning_point))
@printf("Cantera fixed-grid two-point peak strain=%.9g Tmax=%.9g turn_index=%d\n",
    only(ref["turning_strain_rate_per_s"]),only(ref["turning_Tmax_K"]),only(ref["turning_index"]))
@printf("relative peak strain difference=%.6g relative Tmax difference=%.6g\n",
    abs(p.max_strain_rate-only(ref["turning_strain_rate_per_s"]))/only(ref["turning_strain_rate_per_s"]),
    abs(p.max_temperature-only(ref["turning_Tmax_K"]))/only(ref["turning_Tmax_K"]))
@printf("max residual over native curve=%.3g\n",maximum(x.residual_inf for x in curve))

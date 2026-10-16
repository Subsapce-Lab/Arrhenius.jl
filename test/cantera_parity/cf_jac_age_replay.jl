#!/usr/bin/env julia
# Replay Cantera's rejected and accepted counterflow attempts from a captured
# initial snapshot, retaining the native Jacobian workspace between attempts.
using Arrhenius, NPZ, YAML, LinearAlgebra

getvec(npz,key)=vec(npz[key])
function build_diffusion(spec,snapshot,gas)
    z=getvec(snapshot,"flame/grid")
    names=gas.species_names
    Ys=reduce(hcat,[getvec(snapshot,"flame/"*name) for name in names])
    f=CounterflowDiffusionFlame(gas;fuel=spec["fuel"],oxidizer=spec["oxidizer"],
        mdot_fuel=spec["mdot_fuel"],mdot_oxidizer=spec["mdot_oxidizer"],
        T_fuel=spec["T"],T_oxidizer=get(spec,"T_oxidizer",spec["T"]),
        P=spec["P_atm"]*one_atm,width=z[end]-z[1],grid=z)
    f.state[1,:].=getvec(snapshot,"flame/T")./1000
    f.state[2:end-3,:].=permutedims(Ys)
    f.state[end-2,:].=getvec(snapshot,"flame/spreadRate")./Arrhenius._counterflow_Vscale
    f.state[end-1,:].=getvec(snapshot,"flame/Lambda")./Arrhenius._counterflow_Lscale
    f.state[end,:].=getvec(snapshot,"flame/density").*getvec(snapshot,"flame/velocity")
    return f
end
function cantera_refresh_counts(path,n)
    lines=readlines(path); counts=Int[]; current=0
    for line in lines
        if occursin(r"^Timestep \([0-9]+\) dt=",line)
            length(counts)<n && push!(counts,0)
            current=length(counts)
        elseif current>0 && current<=n &&
                (occursin("Re-evaluating Jacobian",line) || occursin("Maximum Jacobian age reached",line))
            counts[current]+=1
        elseif current>0 && occursin(r"Timestep \([0-9]+\) (failed|succeeded)",line)
            current=0
        end
    end
    return counts
end
function first_damping(log)
    m=match(r"Counterflow Newton [0-9]+ .*?damping=([0-9.eE+-]+)",log)
    return m===nothing ? nothing : parse(Float64,m.captures[1])
end
function first_difference(ct, native)
    ct_damp=isempty(ct["damping"]) ? nothing : Float64(first(ct["damping"]))
    nd=native["damping"]
    damping_ok = ct_damp===nothing || nd===nothing ? ct_damp===nd :
        abs(nd-ct_damp)<=0.5*10.0^(floor(log10(abs(ct_damp)))-3)
    return ct["accepted"]==native["accepted"] && ct["jacobians"]==native["jacobians"] && damping_ok,
        damping_ok
end
function main(specpath,capture,outpath)
    spec=YAML.load_file(specpath)
    manifest=YAML.load_file(joinpath(capture,"manifest.json"))
    trace=YAML.load_file(joinpath(capture,"ct_decisions.json"))
    attempts=trace["attempts"]
    ct_refresh=cantera_refresh_counts(joinpath(capture,"ct_decisions.log"),length(attempts))
    length(ct_refresh)==length(attempts) || error("Cantera trace attempt count mismatch")
    gas=CreateSolution(spec["mechanism"]*".yaml")
    snap=npzread(joinpath(capture,manifest["stages"][1]["file"]))
    f=build_diffusion(spec,snap,gas); initial=copy(f.state)
    w=CounterflowWorkspace(f); rows=Any[]
    steady_log=open(joinpath(capture,"native_steady.log"),"w")
    steady_ok=redirect_stdout(steady_log) do
        Arrhenius._counterflow_newton!(f,w;loglevel=3)
    end
    close(steady_log); copyto!(f.state,initial)
    for (i,ct) in enumerate(attempts)
        previous=copy(f.state); logpath=joinpath(capture,"native_attempt_$(lpad(i,2,'0')).log")
        log=open(logpath,"w"); iterations=Ref(0); before=w.jac_evals
        accepted=redirect_stdout(log) do
            Arrhenius._counterflow_newton!(f,w;previous,dt=Float64(ct["dt"]),
                maxiters=20,loglevel=3,iterations)
        end
        close(log)
        native_log=read(logpath,String)
        native=Dict("accepted"=>accepted,"jacobians"=>w.jac_evals-before,
            "damping"=>first_damping(native_log),"iterations"=>iterations[])
        ctrow=Dict("dt"=>ct["dt"],"accepted"=>ct["accepted"],
            "jacobians"=>ct_refresh[i],"damping"=>ct["damping"])
        matched,damping_ok=first_difference(ctrow,native)
        push!(rows,Dict("attempt"=>i,"cantera"=>ctrow,"native"=>native,
            "damping_matches_print_precision"=>damping_ok,"decisions_match"=>matched))
        if accepted
            # An accepted attempt advances the state and clears the retry count.
        else
            copyto!(f.state,previous)
            get(ct,"reset_after",false) && Arrhenius._counterflow_reset_species!(f.state,gas.n_species)
        end
        matched || break
    end
    report=Dict("case"=>get(manifest,"case","unknown"),"initial_steady_failed"=>!steady_ok,
        "replayed_attempts"=>length(rows),"matched_attempts"=>count(r->r["decisions_match"],rows),
        "first_difference"=>(any(!r["decisions_match"] for r in rows) ? length(rows) : nothing),
        "attempts"=>rows)
    open(outpath,"w") do io
        YAML.write(io,report)
    end
    println(YAML.write(report))
end
length(ARGS)==3 || error("usage: cf_jac_age_replay.jl spec.json capture_dir report.yaml")
main(ARGS...)

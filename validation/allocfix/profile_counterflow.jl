using Arrhenius, YAML, LinearAlgebra, Profile
BLAS.set_num_threads(1)
spec = YAML.load_file(ARGS[1])
mech = spec["mechanism"] * ".yaml"
gas = CreateSolution(mech)
P = spec["P_atm"] * one_atm
function runfresh()
    f = CounterflowDiffusionFlame(gas; fuel=spec["fuel"], oxidizer=spec["oxidizer"],
        mdot_fuel=spec["mdot_fuel"], mdot_oxidizer=spec["mdot_oxidizer"],
        T_fuel=spec["T"], T_oxidizer=spec["T"], P, width=spec["width"])
    solve!(f; ratio=spec["ratio"], slope=spec["slope"], curve=spec["curve"],
        prune=get(spec,"prune",0.0))
    f.converged || error("solve did not converge")
    return f
end
warm=runfresh()
println("warm converged=$(warm.converged) points=$(length(warm.grid))")
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1.0 runfresh()
allocs=Profile.Allocs.fetch()
byframe=Dict{String,Int}()
for a in allocs.allocs
    i=findfirst(x->occursin("Arrhenius-allocfix/src/",String(x.file)),a.stacktrace)
    i===nothing && continue
    x=a.stacktrace[i]; k="$(x.file):$(x.line) $(x.func)"
    byframe[k]=get(byframe,k,0)+a.size
end
println("total_bytes=$(sum(a.size for a in allocs.allocs) ) allocation_events=$(length(allocs.allocs))")
for (k,v) in sort!(collect(byframe);by=last,rev=true)[1:min(25,length(byframe))]
    println(v,"\t",k)
end

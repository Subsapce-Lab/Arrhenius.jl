using Arrhenius, YAML, LinearAlgebra, Profile
BLAS.set_num_threads(1)
spec = YAML.load_file(ARGS[1])
mech = spec["mechanism"] * ".yaml"
transport = spec["transport"] == "multicomponent" ? :multicomponent : :mixture_averaged
X = haskey(spec, "X") ? Dict(String(k)=>parse(Float64,v) for (k,v) in split.(split(spec["X"], ","), ":")) : nothing
P = spec["P_atm"] * one_atm
gas = CreateSolution(mech)
function runfresh()
    f = FreeFlame(gas; T=spec["T"], P, X, width=spec["width"], transport_model=transport)
    solve!(f; ratio=spec["ratio"], slope=spec["slope"], curve=spec["curve"])
    f.converged || error("solve did not converge")
    return f
end
warm=runfresh()
println("warm converged=$(warm.converged) points=$(length(warm.grid))")
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1.0 runfresh()
allocs = Profile.Allocs.fetch()
check=runfresh()
workspace=Arrhenius.FlameWorkspace(check)
r=similar(check.state)
Arrhenius.flame_residual!(r,check,check.state,workspace)
residual_bytes=@allocated Arrhenius.flame_residual!(r,check,check.state,workspace)
Arrhenius._flame_jacobian(check,check.state,workspace,r;analytic=false)
jacobian_bytes=@allocated Arrhenius._flame_jacobian(check,check.state,workspace,r;analytic=false)
println("residual_bytes=$residual_bytes jacobian_bytes=$jacobian_bytes")
function report_allocs(label,profile)
    byframe=Dict{String,Int}()
    for a in profile.allocs
        i=findfirst(x->occursin("Arrhenius-allocfix/src/",String(x.file)),a.stacktrace)
        i===nothing && continue
        x=a.stacktrace[i]; k="$(x.file):$(x.line) $(x.func)"
        byframe[k]=get(byframe,k,0)+a.size
    end
    println(label," profile_bytes=",sum((a.size for a in profile.allocs); init=0))
    for (k,v) in sort!(collect(byframe);by=last,rev=true)[1:min(12,length(byframe))]
        println(v,"\t",k)
    end
end
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1.0 Arrhenius.flame_residual!(r,check,check.state,workspace)
report_allocs("residual",Profile.Allocs.fetch())
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1.0 Arrhenius._flame_jacobian(check,check.state,workspace,r;analytic=false)
report_allocs("jacobian",Profile.Allocs.fetch())
byframe = Dict{String,Int}()
for a in allocs.allocs
    fr = findfirst(x -> occursin("Arrhenius-allocfix/src/", String(x.file)), a.stacktrace)
    fr === nothing && continue
    x = a.stacktrace[fr]
    key = "$(x.file):$(x.line) $(x.func)"
    byframe[key] = get(byframe,key,0) + a.size
end
println("total_bytes=$(sum((a.size for a in allocs.allocs); init=0)) allocation_events=$(length(allocs.allocs))")
for (k,v) in sort!(collect(byframe); by=x->last(x), rev=true)[1:min(40,length(byframe))]
    println(v, "\t", k)
end

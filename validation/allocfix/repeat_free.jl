using Arrhenius, YAML, LinearAlgebra
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
GC.gc()
bytes=Int[]; gctimes=Float64[]; times=Float64[]
for i in 1:8
    t=@timed runfresh()
    push!(bytes,t.bytes); push!(gctimes,t.gctime); push!(times,t.time)
    println("repeat=$i bytes=$(t.bytes) gc_seconds=$(t.gctime) seconds=$(t.time)")
end
println("mean_bytes=$(sum(bytes)/length(bytes)) total_gc_seconds=$(sum(gctimes)) mean_gc_seconds=$(sum(gctimes)/length(gctimes)) mean_seconds=$(sum(times)/length(times))")

# Eight revolutions of the published n-dodecane ideal-gas engine calculation.
# Prepare dodecane_IG.yaml + .npz with validation/ic_engine_case.py once.
# julia --project=<SciML environment> example/reactors/ic_engine.jl <mechanism.yaml> [output.csv]
# Solver dependencies: SciMLBase and OrdinaryDiffEqSDIRK.
# CSV profiles use the temperature/crank-angle limits. The companion TOML
# reports integrals from all accepted states and the pressure-work ledger.
include("ic_engine_solver.jl")
include("ic_engine_setup.jl")
using TOML
if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("provide the prepared dodecane_IG.yaml mechanism")
    result = solve_ic_engine(ARGS[1];integrator=native_engine_sdirk,progress=true)
    output = ic_engine_observables(result)
    path = write_ic_engine_csv(get(ARGS,2,"ic_engine.csv"),output,result.gas)
    summary = ic_engine_summary(result,output)
    open(path*".toml","w") do io
        TOML.print(io,summary;sorted=true)
    end
    println(result.integrals)
    println("Integrated ",result.integration_points," accepted states; exported ",length(result.times)," profile states.")
end

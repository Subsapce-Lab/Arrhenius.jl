# Fixed-temperature count reproduction with the Cantera-compatible transient policy.
# Run from the directory containing each case's mechanism and spec:
#   julia --project=<Arrhenius.jl> test/cantera_parity/fixed_t_counts.jl <spec.json>...
using Arrhenius, YAML, LinearAlgebra
BLAS.set_num_threads(1)

function run_case(path)
    spec = YAML.load_file(path)
    gas = CreateSolution(spec["mechanism"] * ".yaml")
    transport = spec["transport"] == "multicomponent" ? :multicomponent : :mixture_averaged
    data = transport == :multicomponent ? MultiTransportData(spec["mechanism"] * ".yaml.multicomponent.npz", gas;
        mechanism=spec["mechanism"] * ".yaml") : nothing
    X = Dict(String(k) => parse(Float64, v) for (k, v) in split.(split(spec["X"], ","), ":"))
    flame = BurnerFlame(gas; mdot=spec["mdot"], T=spec["T"], P=spec["P_atm"] * one_atm,
        X, width=spec["width"], transport_model=transport, multicomponent_data=data,
        discretization=:finite_difference)
    set_temperature_profile!(flame, Float64.(spec["profile_z"]), Float64.(spec["profile_T"]); relative=false)
    stats = Arrhenius.reset!(Arrhenius.solver_stats(flame))
    solve!(flame; ratio=spec["ratio"], slope=spec["slope"], curve=spec["curve"],
        prune=min(get(spec, "prune", 0.0), spec["slope"], spec["curve"]),
        jacobian_mode=:finite_difference, cantera_compatible=true)
    flame.converged || error("$(spec["id"]) did not converge")
    println("case=$(spec["id"]) converged=$(flame.converged) points=$(length(flame.grid)) " *
        "jacobians=$(stats.jacobians) newton=$(stats.newton_iterations) " *
        "steps=$(stats.time_steps) regrids=$(stats.regrids) failures=$(stats.time_step_failures + stats.steady_failures)")
end

isempty(ARGS) && error("provide one or more fixed-case spec paths")
foreach(run_case, ARGS)

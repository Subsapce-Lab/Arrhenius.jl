using Arrhenius, YAML, LinearAlgebra
BLAS.set_num_threads(1)

length(ARGS) == 2 || error("usage: r6_fixed_ablation.jl SPEC ROW[a|b|c|d|e]")
spec = YAML.load_file(ARGS[1])
row = ARGS[2]
row in ("a", "b", "c", "d", "e") || error("unknown ablation row: $row")
gas = CreateSolution(basename(spec["mechanism"]) * ".yaml")
transport = spec["transport"] == "multicomponent" ? :multicomponent : :mixture_averaged
data = transport == :multicomponent ? MultiTransportData(
    basename(spec["mechanism"]) * ".yaml.multicomponent.npz", gas;
    mechanism=basename(spec["mechanism"]) * ".yaml") : nothing
X = Dict(String(k) => parse(Float64, v) for (k, v) in split.(split(spec["X"], ","), ":"))
flame = BurnerFlame(gas; mdot=spec["mdot"], T=spec["T"], P=spec["P_atm"] * one_atm,
    X, width=spec["width"], transport_model=transport, multicomponent_data=data,
    discretization=:finite_difference)
set_temperature_profile!(flame, Float64.(spec["profile_z"]), Float64.(spec["profile_T"]);
    relative=false)
stats = Arrhenius.solver_stats(flame)
Arrhenius.reset!(stats)
common = (; ratio=spec["ratio"], slope=spec["slope"], curve=spec["curve"],
    prune=min(get(spec, "prune", 0.0), spec["slope"], spec["curve"]),
    refine_grid=true, auto=false, max_time_steps=500, loglevel=3)

if row == "a"
    solve!(flame; common..., jacobian_mode=:finite_difference)
elseif row == "b"
    solve!(flame; common..., jacobian_mode=:auto)
elseif row == "c"
    solve!(flame; common..., jacobian_mode=:finite_difference,
        newton_options=NewtonPolicyOptions(cantera_compatible=true))
elseif row == "d"
    solve!(flame; common..., jacobian_mode=:auto,
        newton_options=NewtonPolicyOptions(cantera_compatible=true))
else
    solve!(flame; common..., cantera_compatible=true)
end

println("R6RESULT row=$row case=$(spec["id"]) converged=$(flame.converged) " *
    "jacobians=$(stats.jacobians) newton=$(stats.newton_iterations) " *
    "accepted_steps=$(stats.time_steps) failures=$(stats.time_step_failures + stats.steady_failures) " *
    "regrids=$(stats.regrids) points=$(length(flame.grid))")
flame.converged || error("$(spec["id"]) row $row did not converge")

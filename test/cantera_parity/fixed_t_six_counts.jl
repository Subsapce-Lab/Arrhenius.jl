using Arrhenius, YAML, NPZ, LinearAlgebra
BLAS.set_num_threads(1)

function run_case(spec_path, output_path)
    spec = YAML.load_file(spec_path)
    mechanism = spec["mechanism"] * ".yaml"
    gas = CreateSolution(mechanism)
    transport = spec["transport"] == "multicomponent" ? :multicomponent : :mixture_averaged
    data = transport == :multicomponent ? MultiTransportData(mechanism * ".multicomponent.npz", gas;
        mechanism) : nothing
    X = Dict(String(k) => parse(Float64, v) for (k, v) in split.(split(spec["X"], ","), ":"))
    flame = BurnerFlame(gas; mdot=spec["mdot"], T=spec["T"], P=spec["P_atm"] * one_atm,
        X, width=spec["width"], transport_model=transport, multicomponent_data=data,
        discretization=:finite_difference)
    if spec["kind"] == "fixed"
        set_temperature_profile!(flame, Float64.(spec["profile_z"]), Float64.(spec["profile_T"]); relative=false)
    elseif spec["kind"] != "burner"
        error("unsupported case kind $(spec["kind"])")
    end
    stats = Arrhenius.reset!(Arrhenius.solver_stats(flame))
    solve!(flame; ratio=spec["ratio"], slope=spec["slope"], curve=spec["curve"],
        prune=min(get(spec, "prune", 0.0), spec["slope"], spec["curve"]),
        cantera_compatible=true, loglevel=0)
    flame.converged || error("$(spec["id"]) did not converge")
    names = join(gas.species_names, Char(10))
    NPZ.npzwrite(output_path, Dict("grid" => copy(flame.grid), "T" => temperature(flame),
        "Y" => mass_fractions(flame), "inlet_Y" => copy(flame.inlet_Y),
        "species_names_utf8" => Vector{UInt8}(names),
        "jacobians" => [stats.jacobians], "newton_iterations" => [stats.newton_iterations],
        "time_steps" => [stats.time_steps], "regrids" => [stats.regrids],
        "failures" => [stats.time_step_failures + stats.steady_failures]))
    sumY = maximum(abs.(sum(mass_fractions(flame); dims=1) .- 1))
    mdot_spread = maximum(abs.(flame.state[end, :] .- flame.mass_flux))
    println("ARR case=$(spec["id"]) kind=$(spec["kind"]) points=$(length(flame.grid)) " *
        "jacobians=$(stats.jacobians) newton=$(stats.newton_iterations) steps=$(stats.time_steps) " *
        "regrids=$(stats.regrids) failures=$(stats.time_step_failures + stats.steady_failures) " *
        "sumY_inf=$(sumY) mdot_spread=$(mdot_spread)")
end

length(ARGS) >= 2 && iseven(length(ARGS)) || error("usage: fixed_t_six_counts.jl spec.json output.npz [spec.json output.npz ...]")
for i in 1:2:length(ARGS)
    run_case(ARGS[i], ARGS[i + 1])
end

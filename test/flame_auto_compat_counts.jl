using Arrhenius, YAML, LinearAlgebra

BLAS.set_num_threads(1)

function make_flame(spec)
    mechanism = spec["mechanism"] * ".yaml"
    gas = CreateSolution(mechanism)
    transport = spec["transport"] == "multicomponent" ? :multicomponent : :mixture_averaged
    data = transport == :multicomponent ?
        MultiTransportData(mechanism * ".multicomponent.npz", gas; mechanism) : nothing
    X = Dict(String(k) => parse(Float64, v)
        for (k, v) in split.(split(spec["X"], ","), ":"))
    common = (; T=spec["T"], P=spec["P_atm"] * one_atm, X,
        width=spec["width"], transport_model=transport, multicomponent_data=data,
        discretization=:finite_difference)
    if spec["kind"] == "free"
        return FreeFlame(gas; common...)
    elseif spec["kind"] == "burner"
        return BurnerFlame(gas; mdot=spec["mdot"], common...)
    end
    error("only free and burner specs are supported: $(spec["kind"])")
end

for path in ARGS
    spec = YAML.load_file(path)
    for compatible in (false, true)
        flame = make_flame(spec)
        stats = solver_stats(flame)
        solve!(flame; ratio=spec["ratio"], slope=spec["slope"],
            curve=spec["curve"], prune=min(get(spec, "prune", 0.0),
            spec["slope"], spec["curve"]), cantera_compatible=compatible,
            loglevel=0)
        flame.converged || error("$(spec["id"]) did not converge")
        value = spec["kind"] == "free" ? 100 * flame_speed(flame) : 100 * velocity(flame)[1]
        println("case=$(spec["id"]) mode=$(compatible ? "compat" : "plain") " *
            "jacobians=$(stats.jacobians) newton=$(stats.newton_iterations) " *
            "steps=$(stats.time_steps) steady_failures=$(stats.steady_failures) " *
            "time_step_failures=$(stats.time_step_failures) regrids=$(stats.regrids) " *
            "points=$(length(flame.grid)) speed_cm_s=$(value)")
    end
end

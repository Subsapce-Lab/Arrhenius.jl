using Arrhenius, LinearAlgebra, SHA, YAML

base = joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml")
mktempdir() do dir
    root = YAML.load_file(base)
    h2 = only(filter(s -> s["name"] == "H2", root["species"]))
    h2["thermo"]["data"][2][1] += 1.0
    path = joinpath(dir, "nasa7-large.yaml")
    YAML.write_file(path, root)
    sidecar = Arrhenius.npzread(base * ".npz")
    if haskey(sidecar, "source_sha256_utf8")
        sidecar["source_sha256_utf8"] = collect(codeunits(bytes2hex(SHA.sha256(read(path)))))
    end
    Arrhenius.npzwrite(path * ".npz", sidecar)

    gas = CreateSolution(path)
    X = mole_fractions(gas, "H2:1.1,O2:1,AR:5")
    flame = FreeFlame(gas; X, width=0.03)
    outcome = "converged"
    try
        solve!(flame; jacobian_mode=:auto)
    catch err
        outcome = "failed: " * sprint(showerror, err)
    end
    stats = solver_stats(flame)
    println((fixture="nasa7-large", jacobian_mode=:auto, outcome,
        iterations=stats.newton_iterations, jacobians=stats.jacobians,
        steps=stats.time_steps, time_step_failures=stats.time_step_failures,
        points=length(flame.grid)))
end

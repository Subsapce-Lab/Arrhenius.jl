using Arrhenius, LinearAlgebra, SHA, Test, YAML

function _pathomech_flame_fixture(base, path, delta)
    root = YAML.load_file(base)
    h2 = only(filter(s -> s["name"] == "H2", root["species"]))
    h2["thermo"]["data"][2][1] += delta
    YAML.write_file(path, root)
    npz = Arrhenius.npzread(base * ".npz")
    if haskey(npz, "source_sha256_utf8")
        npz["source_sha256_utf8"] = collect(codeunits(bytes2hex(SHA.sha256(read(path)))))
    end
    Arrhenius.npzwrite(path * ".npz", npz)
    path
end

function _pathomech_flame_counts(path, jacobian_mode)
    gas = CreateSolution(path)
    X = mole_fractions(gas, "H2:1.1,O2:1,AR:5")
    flame = FreeFlame(gas; X, width=0.03)
    solve!(flame; jacobian_mode)
    stats = solver_stats(flame)
    @test flame.converged
    @test stats.newton_iterations > 0
    (iterations=stats.newton_iterations, jacobians=stats.jacobians,
     steps=stats.time_steps, failures=stats.time_step_failures, points=length(flame.grid))
end

@testset "NASA7 midpoint perturbations retain free-flame convergence" begin
    base = joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml")
    mktempdir() do dir
        baseline = _pathomech_flame_counts(base, :finite_difference)
        small = _pathomech_flame_fixture(base, joinpath(dir, "nasa7-small.yaml"), 1e-3)
        large = _pathomech_flame_fixture(base, joinpath(dir, "nasa7-large.yaml"), 1.0)
        small_counts = _pathomech_flame_counts(small, :finite_difference)
        large_counts = _pathomech_flame_counts(large, :finite_difference)
        @info "pathological flame counts" baseline small_counts large_counts
        @test small_counts.iterations <= 5 * baseline.iterations
        @test large_counts.iterations <= 5 * baseline.iterations
    end
end

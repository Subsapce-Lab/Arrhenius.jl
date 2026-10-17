using Arrhenius, Test

@testset "accepted premixed pseudo-time steps" begin
    mechanism = joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml")
    gas = CreateSolution(mechanism)
    X = mole_fractions(gas, "H2:1.1,O2:1,AR:5")
    flame = FreeFlame(gas; X, discretization=:finite_difference)
    solve!(flame)
    @test flame.converged
    @test solver_stats(flame).time_steps > 0
end

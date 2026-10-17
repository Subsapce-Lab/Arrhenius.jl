using Arrhenius, Test

@testset "Cantera-compatible free-flame fixed-temperature anchor" begin
    mechanism = joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml")
    gas = CreateSolution(mechanism)
    X = mole_fractions(gas, "H2:1.1,O2:1,AR:5")
    eq = equilibrate(gas; T=300.0, P=one_atm, X, mode=:HP)
    grid = 0.03 .* [0.0, 0.2, 0.3, 0.6, 0.8, 1.0]
    f = FreeFlame(gas; T=300.0, P=one_atm, X, grid,
        discretization=:finite_difference)

    Arrhenius._fresh_analytic_guess!(f, grid; cantera_compatible=true)
    @test length(f.grid) == length(grid) + 1
    @test isapprox(f.grid[f.anchor], 0.0105; atol=1e-14)
    @test isapprox(f.fixed_temperature, 0.75 * 300.0 + 0.25 * eq.T; atol=1e-10)
    @test isapprox(f.state[1, f.anchor] * 1000, f.fixed_temperature; atol=1e-10)

    Arrhenius._fresh_analytic_guess!(f, grid; cantera_compatible=false)
    @test f.grid == grid
    @test length(f.state[1, :]) == length(grid)
    @test f.fixed_temperature == 1000 * f.state[1, f.anchor]
end

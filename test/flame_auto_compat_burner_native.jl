using Arrhenius, Test

@testset "Cantera-compatible burner auto solve" begin
    mechanism = joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml")
    gas = CreateSolution(mechanism)
    X = mole_fractions(gas, "H2:1.1,O2:1,AR:5")
    grid = 0.03 .* [0.0, 0.2, 0.3, 0.6, 0.8, 1.0]
    burner = BurnerFlame(gas; mdot=0.06, T=300.0, P=one_atm, X, grid,
        discretization=:finite_difference)
    solve!(burner; cantera_compatible=true, refine_grid=false)
    @test burner.converged
    sum_y_error = maximum(abs.(sum(burner.state[2:end-1, :]; dims=1) .- 1))
    mass_flux_error = maximum(abs.(burner.state[end, :] ./ burner.mass_flux .- 1))
    @info "burner invariants" sum_y_error mass_flux_error
    @test sum_y_error < 1e-10
    @test mass_flux_error < 1e-12
end

@testset "Cantera-compatible burner velocity guess" begin
    mechanism = joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml")
    grid = 0.03 .* [0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 1.0]
    for fixed_temperature in (false, true)
        gas = CreateSolution(mechanism)
        burner = BurnerFlame(gas; mdot=0.06, T=300.0, P=one_atm,
            X="H2:1.1,O2:1,AR:5", grid, discretization=:finite_difference)
        fixed_temperature && set_temperature_profile!(burner,
            [0.0, 0.2, 1.0], [300.0, 1500.0, 700.0])
        Arrhenius._fresh_analytic_guess!(burner, grid; cantera_compatible=true)
        u = velocity(burner)
        x = burner.inlet_Y ./ gas.MW
        x ./= sum(x)
        eq = equilibrate(gas; T=burner.inlet_temperature, P=burner.pressure, X=x, mode=:HP)
        u0 = burner.mass_flux / Arrhenius._flame_density(burner,
            burner.inlet_temperature, burner.inlet_Y).density
        u1 = burner.mass_flux / Arrhenius._flame_density(burner, eq.T, eq.Y).density
        for j in eachindex(grid)
            fraction = clamp((grid[j] - grid[1]) / (0.2 * (grid[end] - grid[1])), 0.0, 1.0)
            @test u[j] ≈ u0 + fraction * (u1 - u0) rtol=1e-10
        end
        compatible_state = copy(burner.state)
        Arrhenius._fresh_analytic_guess!(burner, grid; cantera_compatible=false)
        @test maximum(abs.(burner.state[end, :] .- burner.mass_flux)) < 1e-14
        @test maximum(abs.(compatible_state[end, :] .- burner.state[end, :])) > 1e-6
    end
end

using Arrhenius, Test, NPZ

@testset "Cantera fixed-temperature burner final velocity" begin
    reference = NPZ.npzread(joinpath(@__DIR__, "data", "cantera_h2o2_fixed_t_velocity.npz"))
    gas = CreateSolution(joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml"))
    grid = 0.03 .* [0.0, 0.25, 0.5, 0.75, 1.0]
    burner = BurnerFlame(gas; mdot=0.07, T=300.0, P=one_atm,
        X="H2:1.1,O2:1,AR:5", grid, discretization=:finite_difference)
    set_temperature_profile!(burner, [0.0, 0.25, 1.0], [300.0, 1600.0, 300.0])
    solve!(burner; cantera_compatible=true, refine_grid=false)
    @test burner.converged
    @test burner.grid == reference["grid"]
    @test velocity(burner) ≈ reference["velocity"] rtol=2e-5 atol=1e-10
end

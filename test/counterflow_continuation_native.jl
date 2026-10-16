using Test, Arrhenius

@testset "counterflow pseudo-arclength continuation" begin
    mechanism=joinpath(dirname(pathof(Arrhenius)),"..","mechanism","h2o2.yaml")
    if isfile(mechanism*".npz")
        gas=CreateSolution(mechanism)
        grid=collect(range(0,.018;length=12))
        f=CounterflowDiffusionFlame(gas;fuel="H2:1",oxidizer="O2:1",
            mdot_fuel=.5,mdot_oxidizer=3.,T_oxidizer=500.,width=.018,grid)
        solve!(f;refine_grid=false,max_time_steps=100)
        base=(f.fuel_mass_flux,f.oxidizer_mass_flux)
        curve=pseudo_arclength!(f;parameter=:inlet_flux_scale,steps=2,ds=.01,tolerance=1e-7)
        @test length(curve)==3
        @test all(p->p.residual_inf<1e-7,curve)
        @test all(diff([p.arclength for p in curve]).>0)
        @test curve[2].parameter>curve[1].parameter
        @test f.fuel_mass_flux≈base[1]*curve[end].parameter rtol=1e-10
        @test f.oxidizer_mass_flux≈base[2]*curve[end].parameter rtol=1e-10
        @test all(size(p.state)==size(f.state) for p in curve)
        @test all(p->p.max_strain_rate>0,curve)
        @test !any(p.turning_point for p in curve)
    end
end

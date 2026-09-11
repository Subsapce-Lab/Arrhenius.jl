using Test, Arrhenius, LinearAlgebra, NPZ
if !isdefined(Arrhenius,:CounterflowDiffusionFlame)
    Base.include(Arrhenius,joinpath(@__DIR__,"..","src","CounterflowFlames.jl"))
end
@testset "native axisymmetric counterflow" begin
    mechanism=joinpath(dirname(pathof(Arrhenius)),"..","mechanism","h2o2.yaml")
    if isfile(mechanism*".npz")
        gas=CreateSolution(mechanism)
        kwargs=(fuel="H2:1,AR:1",oxidizer="O2:.2,AR:.8",mdot_fuel=.24,mdot_oxidizer=.72)
        @test_throws ArgumentError Arrhenius.CounterflowDiffusionFlame(gas;kwargs...,P=-1)
        @test_throws ArgumentError Arrhenius.CounterflowDiffusionFlame(gas;kwargs...,grid=[0.,.01,.005,.015,.02])
        @test_throws ArgumentError Arrhenius.CounterflowDiffusionFlame(gas;kwargs...,boundary_emissivities=(0.,1.1))
        f=Arrhenius.CounterflowDiffusionFlame(gas;kwargs...)
        solve!(f;slope=.2,curve=.3)
        @test f.converged
        @test 2300<maximum(temperature(f))<2600
        @test temperature(f)[[1,end]] ≈ [300.,300.] atol=1e-6
        @test f.state[end,1] ≈ .24 atol=1e-8
        @test f.state[end,end] ≈ -.72 atol=1e-8
        @test velocity(f)[1]>0 && velocity(f)[end]<0
        @test minimum(mass_fractions(f))>-1e-7
        @test maximum(abs.(sum(mass_fractions(f);dims=1).-1))<1e-10
        @test maximum(abs.(Arrhenius.pressure_curvature(f).-Arrhenius.pressure_curvature(f)[1]))<1e-7
        r=similar(f.state)
        Arrhenius.counterflow_residual!(r,f)
        @test norm(r,Inf)<1e-8
        if haskey(ENV,"COUNTERFLOW_REFERENCE_DIR")
            ref=npzread(joinpath(ENV["COUNTERFLOW_REFERENCE_DIR"],"counterflow-h2-reference.npz"))
            interp(values)=[Arrhenius._interpolate_profile(ref["grid"],vec(values),z) for z in f.grid]
            @test maximum(abs.(temperature(f).-interp(ref["T"])))<1
            @test maximum(abs.(velocity(f).-interp(ref["velocity"])))<1e-3
            @test Arrhenius.pressure_curvature(f)[1] ≈ ref["Lambda"][1] rtol=1e-4
            @test maximum(abs.(mass_fractions(f).-reduce(vcat,permutedims(interp(ref["Y"][k,:])) for k in 1:gas.n_species)))<1e-4
        end
        Tbefore=maximum(temperature(f)); grid=copy(f.grid)
        @test all(iszero,Arrhenius.radiative_heat_loss(f))
        Arrhenius.set_radiation!(f,true;boundary_emissivities=(.3,.7))
        solve!(f;refine_grid=false)
        @test f.converged && f.grid==grid
        @test maximum(temperature(f))<Tbefore
        source=Arrhenius.radiation_source(f)
        @test maximum(source.heat_loss)>0
        @test source.heat_loss[end]==0
        @test maximum(source.planck_absorption)>0
    else
        @test_skip false
    end
end

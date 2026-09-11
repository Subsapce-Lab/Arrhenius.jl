using Test, Arrhenius, YAML, LinearAlgebra, ForwardDiff
isdefined(Arrhenius,:RedlichKwongThermo) || Base.include(Arrhenius,joinpath(@__DIR__,"..","src","RealGasThermo.jl"))

@testset "Native Redlich–Kwong thermodynamic identities" begin
    model = RedlichKwongThermo(joinpath(@__DIR__,"..","mechanism","nDodecane_Reitz.yaml"))
    @test model.geometric_mixing
    @test length(model.MW) == 100
    @test model.b[2] ≈ 0.02208100907
    @test model.a0[2,2] ≈ 1.74102e6
    x = collect(1.:100.)
    x ./= sum(x)
    # Avoid polynomial region boundaries when taking temperature derivatives.
    for (T,P) in ((760.,40one_atm),(1001.,40one_atm),(2000.,100one_atm))
        state = redlich_kwong_state(model;T,P,X=x)
        @test state.h-state.u ≈ P*state.v rtol=1e-12
        @test dot(x,state.partial_molar_enthalpies) ≈ state.h rtol=1e-12
        @test dot(x,state.partial_molar_volumes) ≈ state.v rtol=1e-12
        @test state.cp-state.cv ≈ -T*state.dpdT^2/state.dpdv rtol=1e-12
        du = ForwardDiff.derivative(T) do t
            work = RedlichKwongWorkspace(model,typeof(t))
            redlich_kwong_properties!(work,model,t,state.rho,x).u
        end
        @test du ≈ state.cv rtol=1e-12
        dp = ForwardDiff.derivative(T) do t
            work = RedlichKwongWorkspace(model,typeof(t))
            redlich_kwong_properties!(work,model,t,state.rho,x).P
        end
        @test dp ≈ state.dpdT rtol=1e-12
        # Differentiate total energy with respect to individual mole amounts,
        # keeping total volume and temperature fixed. This independently tests
        # the species energy coefficients required by the reacting ODE.
        utilde = ForwardDiff.gradient(x) do amounts
            total = sum(amounts)
            composition = amounts/total
            density = dot(amounts,model.MW)/state.v
            work = RedlichKwongWorkspace(model,eltype(amounts))
            total*redlich_kwong_properties!(work,model,T,density,composition).u
        end
        @test utilde ≈ state.u_TV rtol=5e-12 atol=1e-6
        # Maxwell relation (ds/dv)_T = (dP/dT)_v.
        dsdv = ForwardDiff.derivative(state.v) do v
            work = RedlichKwongWorkspace(model,typeof(v))
            redlich_kwong_properties!(work,model,T,state.MW/v,x).s
        end
        @test dsdv ≈ state.dpdT rtol=2e-12
        sback = redlich_kwong_state(model;T,rho=state.rho,X=x)
        @test sback.P ≈ P rtol=1e-12
    end
    composition = "c12h26:1,o2:18.5,n2:69.56"
    initial = redlich_kwong_state(model;T=1000.,P=40one_atm,X=composition)
    masses = initial.X.*model.MW
    @test redlich_kwong_state(model;T=1000.,P=40one_atm,X=masses,basis=:mass).rho ≈ initial.rho
    @test redlich_kwong_state(model;T=1000.,P=40one_atm,X=Dict("c12h26"=>1,"o2"=>18.5,"n2"=>69.56)).rho == initial.rho
    dilute = redlich_kwong_state(model;T=1000.,P=1.,X=composition)
    @test abs(dilute.Z-1) < 1e-8
    @test maximum(abs,dilute.lnphi) < 1e-6
    @test abs(dilute.h_departure) < 1
end

@testset "Native RK input and root validation" begin
    path = joinpath(@__DIR__,"..","mechanism","nDodecane_Reitz.yaml")
    model = RedlichKwongThermo(path)
    @test_throws ArgumentError RedlichKwongThermo(path;phase="nDodecane_IG")
    @test_throws ArgumentError RedlichKwongThermo(path;phase="missing")
    @test_throws ArgumentError RedlichKwongThermo(path;molecular_weights=[1.])
    for kwargs in ((T=-1.,P=1e5),(T=1000.,P=-1.),(T=1000.,P=1e5,rho=1.),
                   (T=1000.,rho=-1.),(T=1000.,rho=1e5),(T=1000.,P=1e5,root=:unknown))
        @test_throws Exception redlich_kwong_state(model;X="n2",kwargs...)
    end
    for comp in ("unknown","o2:-1,o2:2",[0. for _ in 1:100],fill(NaN,100),[-1.;ones(99)],ones(99))
        @test_throws Exception redlich_kwong_state(model;T=1000.,P=1e5,X=comp)
    end
    @test_throws ArgumentError redlich_kwong_properties!(RedlichKwongWorkspace(model),model,1000.,1.,ones(100))
    @test_throws ArgumentError redlich_kwong_state(model;T=1000.,P=1e5,X="n2",basis=:unknown)
    # A pure species below its critical temperature has three real cubic roots.
    # Explicit gas/liquid selection preserves the two stable branches.
    yaml = YAML.load_file(path)
    co2 = deepcopy(only(filter(s -> s["name"] == "co2",yaml["species"])))
    doc = Dict("units"=>yaml["units"],"phases"=>[Dict("name"=>"co2","thermo"=>"Redlich-Kwong","species"=>["co2"])],"species"=>[co2])
    pure = RedlichKwongThermo(doc)
    gas = redlich_kwong_state(pure;T=280.,P=4e6,X="co2",root=:gas)
    liquid = redlich_kwong_state(pure;T=280.,P=4e6,X="co2",root=:liquid)
    @test liquid.rho > 3gas.rho
    @test gas.P ≈ liquid.P rtol=1e-10
    @test gas.dpdv < 0 && liquid.dpdv < 0
    # Explicit coefficient units agree with inherited cm/mol units.
    explicit = deepcopy(doc)
    explicit["species"][1]["equation-of-state"]["a"] = "6454490 Pa*m^6*K^0.5/kmol^2"
    explicit["species"][1]["equation-of-state"]["b"] = "0.02965304882 m^3/kmol"
    @test RedlichKwongThermo(explicit).a0 ≈ pure.a0
    @test RedlichKwongThermo(explicit).b ≈ pure.b
    bad = deepcopy(doc)
    bad["species"][1]["equation-of-state"]["b"] = -1
    @test_throws ArgumentError RedlichKwongThermo(bad)
    delete!(bad["species"][1],"equation-of-state")
    @test_throws ArgumentError RedlichKwongThermo(bad)
    database = Dict("species"=>[Dict("name"=>"CO2","critical-parameters"=>
        Dict("critical-temperature"=>304.1282,"critical-pressure"=>7.3773e6))])
    derived = RedlichKwongThermo(bad;critical_properties=database)
    @test derived.b[1] ≈ 0.0866403499650*R*304.1282/7.3773e6
    # Nonzero temperature coefficients and an overridden unlike-species pair.
    binary_doc = deepcopy(yaml)
    binary_doc["phases"] = [Dict("name"=>"binary","thermo"=>"Redlich-Kwong","species"=>["o2","n2"])]
    binary_doc["species"] = deepcopy(filter(s -> s["name"] in ("o2","n2"),yaml["species"]))
    binary_doc["species"][1]["equation-of-state"]["a"][2] = 1.2e8
    binary_doc["species"][2]["equation-of-state"]["a"][2] = 2.3e8
    binary_doc["species"][1]["equation-of-state"]["binary-a"] = Dict("n2"=>[1.3e12,1.7e8])
    binary = RedlichKwongThermo(binary_doc)
    @test !binary.geometric_mixing
    xb = [.21,.79]
    state = redlich_kwong_state(binary;T=700.,P=1e7,X=xb)
    du = ForwardDiff.derivative(700.) do t
        redlich_kwong_properties!(RedlichKwongWorkspace(binary,typeof(t)),binary,t,state.rho,xb).u
    end
    @test du ≈ state.cv rtol=1e-12
    utilde = ForwardDiff.gradient(xb) do amounts
        total = sum(amounts)
        density = dot(amounts,binary.MW)/state.v
        total*redlich_kwong_properties!(RedlichKwongWorkspace(binary,eltype(amounts)),binary,700.,density,amounts/total).u
    end
    @test utilde ≈ state.u_TV rtol=1e-12
    binary_doc["species"][2]["equation-of-state"]["binary-a"] = Dict("o2"=>[1.4e12,1.7e8])
    @test_throws ArgumentError RedlichKwongThermo(binary_doc)
end

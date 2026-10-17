using Arrhenius, LinearAlgebra, Test

@testset "point evaluation parity fixes" begin
    reference = Dict{Tuple{String,String},Float64}()
    for line in eachline(joinpath(@__DIR__, "data", "point_fixes_cantera.csv"))
        fields = split(line, ',')
        if length(fields) == 3 && fields[1] != "cantera_version"
            key = fields[1] in ("reactor_C_H","network_C_H","plog_floor") ?
                string(parse(Float64,fields[2])) : fields[2]
            reference[(fields[1],key)] = parse(Float64,fields[3])
        end
    end
    @test Arrhenius._capped_reverse_rate(1.0, 0.0) == 1e300
    @test Arrhenius._capped_reverse_rate(2.0, 1e-320) == 1e300
    @test isfinite(Arrhenius._capped_reverse_rate(1.0, floatmin(Float64)))
    @test Arrhenius._falloff_reduced_pressure(1.0, 1e-10, 0.0) ≈ 1e290
    @test Arrhenius._falloff_reduced_pressure(1.0, -1e-20, 1.0) < 0

    # Two expressions per pressure exercise Cantera's 1e-300 grouped-rate floor,
    # including its duplicated first/last groups outside the tabulated range.
    plog = Arrhenius.PlogData([1], [0], [1, 3], [one_atm, 10one_atm], [1, 3, 5],
        [1.0 0.0 1e9; 1.0 0.0 1e9; 1.0 0.0 1e9; 1.0 0.0 1e9])
    for pressure in (0.1, 3.0, 100.0)
        expected = reference[("plog_floor",string(pressure))]
        @test expected ≈ 1e-300 rtol=1e-12
        @test Arrhenius._plog_rate(plog, 1, 1.0, pressure*one_atm, 0.0) == expected
    end

    gas = CreateSolution(joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml"))
    h = findfirst(==("H"), gas.species_names)
    ar = findfirst(==("AR"), gas.species_names)
    @test !isnothing(h) && !isnothing(ar)
    reactor = IdealGasReactor(gas; temperature=1400.0, pressure=one_atm,
        mole_fractions=Dict("H2"=>2.0, "O2"=>1.0, "AR"=>4.0))
    rhs = reactor_rhs(reactor)
    base = reactor_state(reactor)
    for y in (0.0, 1e-25, -1e-25, -1e-7)
        u = copy(base)
        u[h] = y
        u[ar] -= y
        du = similar(u)
        rhs(du, u, nothing, 0.0)
        rho = Arrhenius._reactor_tpρ(reactor, u)[3]
        @test rhs.workspace.C[h] ≈ reference[("reactor_C_H",string(y))] rtol=3e-15 atol=1e-40
        @test rhs.workspace.C[h] == y*rho/gas.MW[h]
        @test signbit(rhs.workspace.C[h]) == signbit(y)
    end

    network_reactor = IdealGasReactor(gas; temperature=1400.0, pressure=one_atm,
        mole_fractions=Dict("H2"=>2.0, "O2"=>1.0, "AR"=>4.0),
        constraint=:constant_volume)
    network = ReactorNetwork((vessel=WellStirredReactor(network_reactor; volume=0.2),))
    nrhs = network_rhs(network)
    nbase = network_state(network)
    mass = sum(nbase[1:gas.n_species])
    for y in (0.0, 1e-25, -1e-25, -1e-7)
        u = copy(nbase)
        u[h] = y*mass
        u[ar] += (base[h]-y)*mass
        du = similar(u)
        nrhs(du,u,nothing,0.0)
        state = nrhs.state_vector[1]
        @test nrhs.workspaces[1].C[h] ≈ reference[("network_C_H",string(y))] rtol=3e-15 atol=1e-40
        @test nrhs.workspaces[1].C[h] == y*state.density/gas.MW[h]
    end

    transport = Arrhenius.TransportWorkspace(gas)
    X = zeros(gas.n_species); X[findfirst(==("H2"),gas.species_names)] = 1
    mixture_transport!(transport, gas, one_atm, 1000.0, X; basis=:mole)
    pure_mole = copy(transport.diffusion)
    mixture_transport!(transport, gas, one_atm, 1000.0, X; basis=:mass)
    pure_mass = copy(transport.diffusion)
    h2 = findfirst(==("H2"),gas.species_names)
    # Cantera 4.0a2 returns zero for the dominant species on both bases in a pure state.
    @test pure_mole[h2] == reference[("transport_mix_H2","pure")]
    @test pure_mass[h2] == reference[("transport_mass_H2","pure")]

    X .= 0
    X[h2] = 0.9999999999989999 # exact composition returned by Cantera's TPX transfer
    X[findfirst(==("O2"),gas.species_names)] = 1e-12
    mixture_transport!(transport, gas, one_atm, 1000.0, X; basis=:mole)
    mole_diffusion = copy(transport.diffusion)
    mixture_transport!(transport, gas, one_atm, 1000.0, X; basis=:mass)
    mass_diffusion = copy(transport.diffusion)
    Xf = max.(X,1e-20)
    @test mole_diffusion[h2] ≈ reference[("transport_mix_H2","near")] rtol=2e-13
    @test mass_diffusion[h2] ≈ reference[("transport_mass_H2","near")] rtol=2e-13
    @test reference[("transport_mix_H2","pure")] == 0.0
    @test reference[("transport_mass_H2","pure")] == 0.0
    for k in eachindex(X)
        sum1 = sum(Xf[j]*transport.inverse_binary[j,k] for j in eachindex(X) if j != k)
        sum2 = sum(Xf[j]*gas.MW[j]*transport.inverse_binary[j,k] for j in eachindex(X) if j != k)
        meanmw = dot(X,gas.MW)
        othermass = meanmw-Xf[k]*gas.MW[k]
        expected_mole = sum1 > 0 ? othermass/(meanmw*sum1) : transport.binary[k,k]
        expected_mass = sum1 > 0 && othermass > 0 ?
            inv(sum1+Xf[k]*sum2/othermass) : transport.binary[k,k]
        @test mole_diffusion[k] ≈ expected_mole rtol=2e-14
        @test mass_diffusion[k] ≈ expected_mass rtol=2e-14
    end
end

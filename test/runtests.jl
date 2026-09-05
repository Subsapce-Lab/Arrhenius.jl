using Arrhenius
using ForwardDiff
using LinearAlgebra
using SHA
using Test

include("zero_collider.jl")
include("zero_collider_partials.jl")
include("plog_partials.jl")
include("constant_pressure_jacobian.jl")

@testset "sidecar provenance" begin
    mechanism, stream = mktemp()
    write(stream, "mechanism fixture\n")
    close(stream)
    digest = bytes2hex(SHA.sha256(read(mechanism)))
    metadata = Dict(
        "sidecar_format_utf8" => collect(codeunits("arrhenius-sidecar-v2")),
        "source_sha256_utf8" => collect(codeunits(digest)),
    )
    @test Arrhenius._validate_sidecar_metadata(metadata, mechanism) === nothing
    metadata_v3 = copy(metadata)
    metadata_v3["sidecar_format_utf8"] = collect(codeunits("arrhenius-sidecar-v3"))
    @test Arrhenius._validate_sidecar_metadata(metadata_v3, mechanism) === nothing

    stale = copy(metadata)
    stale["source_sha256_utf8"] = collect(codeunits(repeat("0", 64)))
    @test_throws ArgumentError Arrhenius._validate_sidecar_metadata(stale, mechanism)

    unknown = copy(metadata)
    unknown["sidecar_format_utf8"] = collect(codeunits("arrhenius-sidecar-v99"))
    @test_throws ArgumentError Arrhenius._validate_sidecar_metadata(unknown, mechanism)
end

@testset "optional sidecar reaction-family indices" begin
    @test Arrhenius._sidecar_reaction_family_indices(Dict()) === nothing
    indices = Arrhenius._sidecar_reaction_family_indices(Dict(
        "index_falloff" => [4, 9],
        "index_falloff_Troe" => [1, -1],
    ))
    @test indices == (Int64[], Int64[4, 9], Int64[1, -1])
    @test_throws ArgumentError Arrhenius._sidecar_reaction_family_indices(Dict(
        "index_falloff" => [4],
    ))
end

@testset "single-region NASA7" begin
    yaml = Dict(
        "phases" => [Dict(
            "species" => ["A", "B"],
            "elements" => ["X"],
        )],
        "reactions" => Any[],
        "species" => [
            Dict(
                "name" => "A",
                "composition" => Dict("X" => 1),
                "thermo" => Dict(
                    "temperature-ranges" => [200.0, 3000.0],
                    "data" => [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
                ),
            ),
            Dict(
                "name" => "B",
                "composition" => Dict("X" => 1),
                "thermo" => Dict(
                    "temperature-ranges" => [200.0, 800.0, 3000.0],
                    "data" => [
                        [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
                    ],
                ),
            ),
        ],
    )
    thermo = Arrhenius.IdealGasThermo(yaml)
    @test thermo.nasa_low[1, :] == thermo.nasa_high[1, :]
    @test thermo.Trange[1, :] == [200.0, 1000.0, 3000.0]
    @test thermo.Trange[2, :] == [200.0, 800.0, 3000.0]
end

@testset "PLOG interpolation and duplicate rates" begin
    plog = Arrhenius.PlogData(
        [7],
        [2],
        [1, 3],
        [1.0e5, 1.0e6],
        [1, 3, 4],
        [2.0 0.0 0.0; 3.0 0.0 0.0; 20.0 0.0 0.0],
    )
    T = 1000.0
    @test Arrhenius._plog_rate(plog, 1, T, 1.0e4, log(T)) ≈ 5.0
    @test Arrhenius._plog_rate(plog, 1, T, sqrt(1.0e11), log(T)) ≈ 10.0
    @test Arrhenius._plog_rate(plog, 1, T, 1.0e7, log(T)) ≈ 20.0
    concentrations = [3.0, 0.25]
    @test Arrhenius._plog_rate_with_collider(
        plog, 1, T, 1.0e4, log(T), concentrations,
    ) ≈ 1.25
    legacy_plog = Arrhenius.PlogData(
        [7], [1, 3], [1.0e5, 1.0e6], [1, 3, 4],
        [2.0 0.0 0.0; 3.0 0.0 0.0; 20.0 0.0 0.0],
    )
    @test legacy_plog.collider_indices == [0]
    plog32 = Arrhenius.PlogData(
        [7], [2], [1, 3], Float32[1.0e5, 1.0e6], [1, 3, 4],
        Float32[2.0 0.0 0.0; 3.0 0.0 0.0; 20.0 0.0 0.0],
    )
    T32 = 1000.0f0
    @test eltype(plog32.Arrhenius_coeffs) == Float32
    @test Arrhenius._plog_rate(plog32, 1, T32, 1.0f4, log(T32)) ≈ 5.0f0
    log_rates = LogRateData(
        Float32[], Float32[], Float32[],
        Float32[], Float32[], Float32[],
        Float32.(log.([2.0, 3.0, 20.0])),
        zeros(Float32, 3),
        zeros(Float32, 3),
    )
    @test Arrhenius._plog_rate(
        log_rates,
        plog,
        1,
        T,
        sqrt(1.0e11),
        log(T32),
        Float32(4184.0 / R) / T32,
    ) ≈ 10.0 rtol=1.0e-5
end

@testset "backend-neutral ensemble API" begin
    mechanism = MechanismIR(
        species_names=["fuel", "product"],
        molecular_weights=[17.0, 18.0],
        n_reactions=3,
        packed=(reaction_ids=collect(1:3),),
        metadata=(name="synthetic",),
    )
    @test mechanism.n_species == 2
    @test MechanismIR(mechanism) === mechanism
    @test_throws DimensionMismatch MechanismIR(
        species_names=["fuel"],
        molecular_weights=[17.0, 18.0],
        n_reactions=1,
        packed=(;),
    )
    constant_volume = HomogeneousIdealGasReactor()
    @test reactor_contract(constant_volume) == (
        model=Symbol("adiabatic-closed-constant-volume-ideal-gas"),
        state_variables=(:mass_fractions, :temperature_K),
    )
    @test reactor_contract(HomogeneousIdealGasReactor(
        constraint=:constant_pressure,
    )).model == Symbol("adiabatic-closed-constant-pressure-ideal-gas")
    @test_throws ArgumentError HomogeneousIdealGasReactor(constraint=:isobaric)
    @test_throws ArgumentError HomogeneousIdealGasReactor(
        state_variables=(:mole_fractions, :temperature_K),
    )

    perturbation = SparseLogAPerturbations([2], [log(3.0)])
    multipliers = materialize_rate_multipliers(perturbation, 3, Float32)
    @test eltype(multipliers) == Float32
    @test multipliers ≈ Float32[1.0, 3.0, 1.0]
    @test_throws ArgumentError SparseLogAPerturbations([1, 1], [0.0, 1.0])

    samples = [
        EnsembleSample(
            20,
            (forcing=2.0,);
            perturbations=perturbation,
            metadata=(group=:low,),
        ),
        EnsembleSample(
            10,
            (forcing=4.0,);
            perturbations=SparseLogAPerturbations([2], [log(2.0)]),
            metadata=(group=:high,),
        ),
    ]
    qoi = QoISchema(
        (response=(payload, sample) -> payload.base + sample.id / 100,);
        units=(response=:dimensionless,),
    )
    manifest = EnsembleManifest(
        samples;
        qoi=qoi,
        metadata=(campaign="unit-test",),
    )
    reactor = (kind=:synthetic, gain=2.0)
    precision = PrecisionPolicy(rates=Float32, qoi=Float64)
    sample_solver = function (ir, reactor, sample, policies)
        local_multipliers = materialize_rate_multipliers(
            sample.perturbations,
            ir.n_reactions,
            policies.precision.rates,
        )
        return (
            base=reactor.gain * sample.state.forcing * local_multipliers[2],
        )
    end
    prepared = prepare_ensemble(
        mechanism,
        reactor;
        precision_policy=precision,
        jacobian=FiniteDifferenceJacobian(1.0e-6),
        linear_solver=SparseLinearSolver(ordering=:natural),
        sample_solver=sample_solver,
    )
    @test prepared isa PreparedEnsemble
    @test backend_status(prepared.backend) == :available
    result_type = typeof(QoIResult(0, (response=0.0,)))
    results = Vector{result_type}(undef, length(samples))
    @test solve_ensemble!(
        results,
        prepared,
        manifest;
        batch_policy=1,
    ) === results
    @test getfield.(results, :sample_id) == [20, 10]
    @test results[1].values.response ≈ 12.2
    @test results[2].values.response ≈ 16.1

    threaded = prepare_ensemble(
        mechanism,
        reactor;
        backend=CPUBackend(threaded=true),
        precision_policy=precision,
        sample_solver=sample_solver,
    )
    threaded_results = similar(results)
    solve_ensemble!(threaded_results, threaded, manifest; batch_policy=:auto)
    @test threaded_results == results

    profile = profile_ensemble(
        prepared,
        manifest;
        repetitions=2,
        warmup=false,
        batch_policy=1,
    )
    @test profile.sample_count == 2
    @test profile.repetitions == 2
    @test profile.batch_policy == 1
    @test profile.effective_batch_size == 1
    @test length(profile.elapsed_seconds) == 2
    @test profile.median_seconds >= 0.0
    @test profile.samples_per_second > 0.0
    @test profile.results == results

    raw_manifest = EnsembleManifest(samples)
    raw_prepared = prepare_ensemble(
        mechanism,
        reactor;
        sample_solver=(ir, reactor, sample, policies) ->
            (raw=reactor.gain * sample.id,),
    )
    raw_results = Vector{typeof(QoIResult(0, (raw=0.0,)))}(undef, 2)
    solve_ensemble!(raw_results, raw_prepared, raw_manifest)
    @test raw_results[1].values == (raw=40.0,)

    out_of_range = EnsembleManifest([
        EnsembleSample(
            1,
            (forcing=1.0,);
            perturbations=SparseLogAPerturbations([4], [0.1]),
        ),
    ])
    out_of_range_results = Vector{Any}(undef, 1)
    @test_throws ArgumentError solve_ensemble!(
        out_of_range_results,
        prepared,
        out_of_range,
    )
    @test_throws ArgumentError solve_ensemble!(
        results,
        prepared,
        manifest;
        batch_policy=0,
    )
    @test_throws ArgumentError solve_ensemble!(
        results,
        prepared,
        manifest;
        batch_policy=:fixed,
    )
    @test_throws ArgumentError solve_ensemble!(
        results,
        prepared,
        manifest;
        batch_policy=true,
    )
    @test_throws ArgumentError prepare_ensemble(mechanism, reactor)

    @test backend_status(CUDABackend()) == :extension_not_loaded
    @test backend_status(MetalBackend()) == :extension_not_loaded
    @test_throws BackendUnavailableError prepare_ensemble(
        mechanism,
        reactor;
        backend=CUDABackend(),
        sample_solver=sample_solver,
    )
    @test_throws BackendUnavailableError prepare_ensemble(
        mechanism,
        reactor;
        backend=MetalBackend(),
        sample_solver=sample_solver,
    )
end

@testset "jl" begin
    # Write your tests here.

    mechanism_path = joinpath(@__DIR__, "..", "mechanism", "gri30.yaml")
    gas = CreateSolution(mechanism_path)
    gas_ir = MechanismIR(gas)
    @test gas_ir.n_species == gas.n_species
    @test gas_ir.packed.reaction === gas.reaction
    ns = gas.n_species

    Y0 = ones(ns) ./ ns
    T0 = 1200.0
    P = one_atm
    wdot = set_states(gas, T0, P, Y0)
    @show size(wdot)
    @show wdot[1:3]

    mean_MW = 1.0 / dot(Y0, 1 ./ gas.MW)
    density = P / R / T0 * mean_MW
    X = Y2X(gas, Y0, mean_MW)
    C = Y2C(gas, Y0, density)
    h_mole = get_H(gas, T0, Y0, X)
    S0 = get_S(gas, T0, P, X)
    thermo_buffer = zeros(ns)
    @test cal_h_RT!(thermo_buffer, gas, T0, P, X) ≈ cal_h_RT(gas, T0, P, X)
    @test cal_s0_R!(thermo_buffer, gas, T0, P, X) ≈ cal_s0_R(gas, T0, P, X)
    @test cal_cp_R!(thermo_buffer, gas, T0, P, X) ≈ cal_cp_R(gas, T0, P, X)
    kinetics_workspace = KineticsWorkspace(gas.reaction)
    wdot_buffer = zeros(ns)
    @test wdot!(
        wdot_buffer, gas.reaction, T0, C, S0, h_mole, kinetics_workspace,
    ) ≈ wdot
    @test_throws ArgumentError convert_precision(gas, Float32)
    baseline_qdot = copy(wdot!(
        wdot_buffer,
        gas.reaction,
        T0,
        C,
        S0,
        h_mole,
        kinetics_workspace;
        get_qdot=true,
    ))
    log_rate_data = LogRateData(gas.reaction, Float32)
    mixed_qdot = copy(wdot!(
        wdot_buffer,
        gas.reaction,
        T0,
        C,
        S0,
        h_mole,
        kinetics_workspace;
        get_qdot=true,
        log_rate_data=log_rate_data,
    ))
    mixed_wdot = gas.reaction.vk * mixed_qdot
    @test all(isfinite, mixed_wdot)
    @test norm(mixed_wdot - wdot, Inf) / norm(wdot, Inf) < 1.0e-3
    rate_multipliers = ones(gas.n_reactions)
    rate_multipliers[1] = 2.0
    perturbed_qdot = wdot!(
        wdot_buffer,
        gas.reaction,
        T0,
        C,
        S0,
        h_mole,
        kinetics_workspace;
        get_qdot=true,
        rate_multipliers=rate_multipliers,
    )
    @test perturbed_qdot[1] ≈ 2.0 * baseline_qdot[1]
    @test_throws DimensionMismatch wdot!(
        wdot_buffer,
        gas.reaction,
        T0,
        C,
        S0,
        h_mole,
        kinetics_workspace;
        rate_multipliers=ones(gas.n_reactions - 1),
    )

    @testset "exact reaction-rate partials" begin
        @test supports_exact_rate_partials(gas.reaction)
        derivative_workspace = KineticsDerivativeWorkspace(gas.reaction)
        qdot = zeros(gas.n_reactions)
        dqdot_dC = zeros(gas.n_reactions, gas.n_species)
        dqdot_dT = zeros(gas.n_reactions)
        reaction_rate_partials!(
            qdot,
            dqdot_dC,
            dqdot_dT,
            gas.reaction,
            T0,
            C,
            S0,
            h_mole,
            derivative_workspace,
        )
        function progress_from_state(state)
            temperature = state[end]
            concentrations = state[1:end-1]
            dual_output = similar(concentrations)
            dual_X = similar(concentrations)
            total_concentration = sum(concentrations)
            dual_X .= concentrations ./ total_concentration
            dual_h = similar(concentrations)
            dual_s = similar(concentrations)
            cal_h_RT!(dual_h, gas, temperature, P, dual_X)
            cal_s0_R!(dual_s, gas, temperature, P, dual_X)
            dual_h .*= R * temperature
            dual_s .*= R
            return wdot_func(
                gas.reaction,
                temperature,
                concentrations,
                dual_s,
                dual_h;
                get_qdot=true,
            )
        end
        positive_concentrations = max.(C, 1.0e-12)
        positive_state = vcat(positive_concentrations, T0)
        exact = hcat(dqdot_dC, dqdot_dT)
        reference = ForwardDiff.jacobian(progress_from_state, positive_state)
        reaction_rate_partials!(
            qdot,
            dqdot_dC,
            dqdot_dT,
            gas.reaction,
            T0,
            positive_concentrations,
            S0,
            h_mole,
            derivative_workspace,
        )
        exact = hcat(dqdot_dC, dqdot_dT)
        @test norm(qdot - progress_from_state(positive_state), Inf) /
            max(norm(qdot, Inf), eps()) < 1.0e-12
        @test norm(exact - reference, Inf) /
            max(norm(reference, Inf), eps()) < 2.0e-10

        positive_Y = max.(Y0, 1.0e-12)
        positive_Y ./= sum(positive_Y)
        constant_volume_state = vcat(positive_Y, T0)
        constant_volume_workspace = ConstantVolumeJacobianWorkspace(gas)
        constant_volume_rhs = similar(constant_volume_state)
        constant_volume_jacobian = zeros(
            length(constant_volume_state),
            length(constant_volume_state),
        )
        constant_volume_jacobian!(
            constant_volume_jacobian,
            constant_volume_rhs,
            gas,
            density,
            constant_volume_state,
            constant_volume_workspace,
        )
        function constant_volume_rhs_reference(state)
            temperature = state[end]
            mass_fractions = state[1:end-1]
            mean_molecular_weight = inv(dot(mass_fractions, 1 ./ gas.MW))
            concentrations = density .* mass_fractions ./ gas.MW
            mole_fractions =
                mass_fractions .* mean_molecular_weight ./ gas.MW
            pressure = density * R * temperature / mean_molecular_weight
            cp_R = cal_cp_R(
                gas,
                temperature,
                pressure,
                mole_fractions,
            )
            enthalpies = R * temperature .* cal_h_RT(
                gas,
                temperature,
                pressure,
                mole_fractions,
            )
            entropies = R .* cal_s0_R(
                gas,
                temperature,
                pressure,
                mole_fractions,
            )
            source = wdot_func(
                gas.reaction,
                temperature,
                concentrations,
                entropies,
                enthalpies,
            )
            cv_mass = R * dot(
                mass_fractions ./ gas.MW,
                cp_R .- one(temperature),
            )
            internal_energy_source = dot(
                enthalpies .- R * temperature,
                source,
            )
            return vcat(
                gas.MW .* source ./ density,
                -internal_energy_source / (density * cv_mass),
            )
        end
        constant_volume_reference =
            constant_volume_rhs_reference(constant_volume_state)
        constant_volume_ad_jacobian = ForwardDiff.jacobian(
            constant_volume_rhs_reference,
            constant_volume_state,
        )
        @test norm(constant_volume_rhs - constant_volume_reference, Inf) /
            max(norm(constant_volume_reference, Inf), eps()) < 1.0e-12
        @test norm(
            constant_volume_jacobian - constant_volume_ad_jacobian,
            Inf,
        ) / max(norm(constant_volume_ad_jacobian, Inf), eps()) < 2.0e-10
    end
    function first_rate_of_progress(multiplier)
        multipliers = ones(typeof(multiplier), gas.n_reactions)
        multipliers[1] = multiplier
        return wdot_func(
            gas.reaction,
            T0,
            C,
            S0,
            h_mole;
            get_qdot=true,
            rate_multipliers=multipliers,
        )[1]
    end
    @test ForwardDiff.derivative(first_rate_of_progress, 1.0) ≈ baseline_qdot[1]

    X_buffer = similar(Y0)
    Y2X!(X_buffer, gas, Y0)
    @test @allocated(Y2X!(X_buffer, gas, Y0)) == 0

    u0 = vcat(Y0, T0)
    function f(u)
        T = u[end]
        Y = @view(u[1:ns])
        mean_MW = 1.0 / dot(Y, 1 ./ gas.MW)
        ρ_mass = P / R / T * mean_MW
        X = Y2X(gas, Y, mean_MW)
        C = Y2C(gas, Y, ρ_mass)
        cp_mole, cp_mass = get_cp(gas, T, X, mean_MW)
        h_mole = get_H(gas, T, Y, X)
        S0 = get_S(gas, T, P, X)
        wdot = wdot_func(gas.reaction, T, C, S0, h_mole)
        Tdot = -dot(h_mole, wdot) / ρ_mass / cp_mass
        return Tdot
    end
    grad = ForwardDiff.gradient(f, u0)
    @show size(grad)
    @show grad[1:3]

end

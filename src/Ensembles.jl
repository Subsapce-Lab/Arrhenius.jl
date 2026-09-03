"""Backend-neutral, typed representation of a chemical mechanism.

`packed` contains the arrays and auxiliary objects consumed by a sample solver
or hardware extension. The `Solution` constructor is intentionally shallow for
the large mechanism arrays, so preparing an ensemble does not duplicate them.
"""
struct MechanismIR{T<:AbstractFloat,P,M<:NamedTuple}
    n_species::Int
    n_reactions::Int
    species_names::Vector{String}
    molecular_weights::Vector{T}
    packed::P
    metadata::M
end

function _validate_mechanism_ir(
    species_names,
    molecular_weights,
    n_reactions,
)
    isempty(species_names) && throw(ArgumentError(
        "a mechanism must contain at least one species",
    ))
    length(species_names) == length(molecular_weights) ||
        throw(DimensionMismatch(
            "species_names and molecular_weights must have the same length",
        ))
    length(unique(species_names)) == length(species_names) ||
        throw(ArgumentError("species names must be unique"))
    all(weight -> isfinite(weight) && weight > zero(weight), molecular_weights) ||
        throw(ArgumentError("molecular weights must be finite and positive"))
    n_reactions >= 0 ||
        throw(ArgumentError("n_reactions must be nonnegative"))
    return nothing
end

"""
    MechanismIR(; species_names, molecular_weights, n_reactions, packed,
                metadata=(;))

Construct an intermediate representation from explicitly packed input.
`packed` should be a concrete object, normally a `NamedTuple` of host arrays.
No layout is imposed by the core package; the selected sample solver or backend
owns that contract.
"""
function MechanismIR(;
    species_names::AbstractVector{<:AbstractString},
    molecular_weights::AbstractVector{T},
    n_reactions::Integer,
    packed,
    metadata::NamedTuple=(;),
) where {T<:AbstractFloat}
    names = String.(species_names)
    weights = collect(molecular_weights)
    _validate_mechanism_ir(names, weights, n_reactions)
    return MechanismIR(
        length(names),
        Int(n_reactions),
        names,
        weights,
        packed,
        metadata,
    )
end

"""
    MechanismIR(gas::Solution; metadata=(;))

Wrap an Arrhenius `Solution` in the backend-neutral representation. Reaction,
thermodynamic, transport, and element data are retained in `ir.packed` without
copying their large arrays.
"""
function MechanismIR(gas::Solution; metadata::NamedTuple=(;))
    packed = (
        reaction=gas.reaction,
        thermo=gas.thermo,
        transport=gas.trans,
        elements=gas.elements,
        element_matrix=gas.ele_matrix,
    )
    return MechanismIR(
        species_names=gas.species_names,
        molecular_weights=gas.MW,
        n_reactions=gas.n_reactions,
        packed=packed,
        metadata=metadata,
    )
end

"""
    MechanismIR(mechanism_file::AbstractString; metadata=(;))

Load an Arrhenius YAML mechanism and its sidecar, then construct a
`MechanismIR`. The source path is not retained in the public object.
"""
function MechanismIR(mechanism_file::AbstractString; metadata::NamedTuple=(;))
    return MechanismIR(CreateSolution(mechanism_file); metadata=metadata)
end

MechanismIR(ir::MechanismIR; metadata::NamedTuple=ir.metadata) =
    metadata === ir.metadata ? ir : MechanismIR(
        species_names=ir.species_names,
        molecular_weights=ir.molecular_weights,
        n_reactions=ir.n_reactions,
        packed=ir.packed,
        metadata=metadata,
    )

"""Sparse, one-based perturbations to reaction pre-exponential factors.

For reaction `i`, `delta_logA[j]` represents a natural-log perturbation and is
equivalent to multiplying its rate by `exp(delta_logA[j])`. Reaction indices
must be unique; they are sorted during construction.
"""
struct SparseLogAPerturbations{T<:AbstractFloat}
    reaction_indices::Vector{Int}
    delta_logA::Vector{T}
    function SparseLogAPerturbations(
        reaction_indices::AbstractVector{<:Integer},
        delta_logA::AbstractVector{T},
    ) where {T<:AbstractFloat}
        length(reaction_indices) == length(delta_logA) ||
            throw(DimensionMismatch(
                "reaction_indices and delta_logA must have the same length",
            ))
        all(>(0), reaction_indices) || throw(ArgumentError(
            "reaction indices use one-based positive indexing",
        ))
        all(isfinite, delta_logA) ||
            throw(ArgumentError("logA perturbations must be finite"))
        order = sortperm(reaction_indices)
        indices = Int.(reaction_indices[order])
        length(unique(indices)) == length(indices) ||
            throw(ArgumentError("reaction indices must be unique"))
        return new{T}(indices, collect(delta_logA[order]))
    end
end

SparseLogAPerturbations(::Type{T}=Float64) where {T<:AbstractFloat} =
    SparseLogAPerturbations(Int[], T[])

"""
    materialize_rate_multipliers(perturbations, n_reactions, T=Float64)

Expand sparse log-domain perturbations into the dense multiplicative vector
accepted by `wdot!` and `wdot_func`.
"""
function materialize_rate_multipliers(
    perturbations::SparseLogAPerturbations,
    n_reactions::Integer,
    ::Type{T}=Float64,
) where {T<:AbstractFloat}
    n_reactions >= 0 ||
        throw(ArgumentError("n_reactions must be nonnegative"))
    if !isempty(perturbations.reaction_indices) &&
       last(perturbations.reaction_indices) > n_reactions
        throw(ArgumentError(
            "a logA perturbation refers to reaction " *
            "$(last(perturbations.reaction_indices)), but the mechanism has " *
            "$n_reactions reactions",
        ))
    end
    multipliers = ones(T, n_reactions)
    @inbounds for (index, delta) in zip(
        perturbations.reaction_indices,
        perturbations.delta_logA,
    )
        multiplier = exp(T(delta))
        isfinite(multiplier) || throw(ArgumentError(
            "logA perturbation at reaction $index overflows in $T",
        ))
        multipliers[index] = multiplier
    end
    return multipliers
end

"""One typed ensemble sample.

`state` is a `NamedTuple` chosen by the application, for example
`(temperature=1000.0, pressure=one_atm, mass_fractions=Y)`. This keeps the core
independent of a particular reactor model while preserving a concrete schema.
"""
struct EnsembleSample{I,S<:NamedTuple,P<:SparseLogAPerturbations,M<:NamedTuple}
    id::I
    state::S
    perturbations::P
    metadata::M
end

function EnsembleSample(
    id,
    state::NamedTuple;
    perturbations::SparseLogAPerturbations=SparseLogAPerturbations(),
    metadata::NamedTuple=(;),
)
    return EnsembleSample(id, state, perturbations, metadata)
end

"""Typed QoI schema.

`reducers` is a named tuple of callables. Each callable receives
`(sample_payload, sample)` and returns one quantity of interest. Optional units
use the same names or a subset of them.
"""
struct QoISchema{R<:NamedTuple,U<:NamedTuple}
    reducers::R
    units::U
end

function QoISchema(reducers::NamedTuple; units::NamedTuple=(;))
    all(name -> name in keys(reducers), keys(units)) ||
        throw(ArgumentError("QoI unit names must also appear in reducers"))
    return QoISchema{typeof(reducers),typeof(units)}(reducers, units)
end

QoISchema(; reducers...) = QoISchema((; reducers...))

"""Homogeneous collection of typed samples, their QoI schema, and metadata."""
struct EnsembleManifest{S<:EnsembleSample,Q<:QoISchema,M<:NamedTuple}
    samples::Vector{S}
    qoi::Q
    metadata::M
end


function EnsembleManifest(
    samples::AbstractVector{S};
    qoi::QoISchema=QoISchema(),
    metadata::NamedTuple=(;),
) where {S<:EnsembleSample}
    return EnsembleManifest{S,typeof(qoi),typeof(metadata)}(
        collect(samples),
        qoi,
        metadata,
    )
end

"""Result values for one ensemble sample."""
struct QoIResult{I,V<:NamedTuple}
    sample_id::I
    values::V
end

"""Floating-point roles advertised to a sample solver or backend."""
struct PrecisionPolicy{
    S<:AbstractFloat,
    R<:AbstractFloat,
    L<:AbstractFloat,
    Q<:AbstractFloat,
}
    state::Type{S}
    rates::Type{R}
    linear::Type{L}
    qoi::Type{Q}
end

function PrecisionPolicy(;
    state::Type{S}=Float64,
    rates::Type{R}=state,
    linear::Type{L}=state,
    qoi::Type{Q}=Float64,
) where {
    S<:AbstractFloat,
    R<:AbstractFloat,
    L<:AbstractFloat,
    Q<:AbstractFloat,
}
    return PrecisionPolicy{S,R,L,Q}(state, rates, linear, qoi)
end

abstract type AbstractJacobianPolicy end

"""Finite-difference Jacobian policy with a relative perturbation size."""
struct FiniteDifferenceJacobian{T<:AbstractFloat} <: AbstractJacobianPolicy
    relative_step::T
    function FiniteDifferenceJacobian(relative_step::T) where {T<:AbstractFloat}
        isfinite(relative_step) && relative_step > zero(T) ||
            throw(ArgumentError("relative_step must be finite and positive"))
        return new{T}(relative_step)
    end
end

FiniteDifferenceJacobian() = FiniteDifferenceJacobian(sqrt(eps(Float64)))

"""User-provided Jacobian callback policy.

The ensemble core does not invoke this callback. A sample solver may use it,
which keeps analytic or automatic Jacobians optional rather than mandatory.
"""
struct ProvidedJacobian{F} <: AbstractJacobianPolicy
    evaluate!::F
end

abstract type AbstractLinearSolverPolicy end

"""Request a dense linear solve from the selected sample solver."""
struct DenseLinearSolver <: AbstractLinearSolverPolicy end

"""Request a sparse linear solve and record its ordering strategy."""
struct SparseLinearSolver <: AbstractLinearSolverPolicy
    ordering::Symbol
end

SparseLinearSolver(; ordering::Symbol=:amd) = SparseLinearSolver(ordering)

"""User-provided linear-solve callback policy."""
struct ProvidedLinearSolver{F} <: AbstractLinearSolverPolicy
    solve!::F
end

"""Precision, Jacobian, and linear-solver policies passed to each solve."""
struct EnsemblePolicies{
    P<:PrecisionPolicy,
    J<:AbstractJacobianPolicy,
    L<:AbstractLinearSolverPolicy,
}
    precision::P
    jacobian::J
    linear_solver::L
end

abstract type AbstractEnsembleBackend end

"""Host CPU backend. Threaded execution requires a thread-safe sample solver."""
struct CPUBackend <: AbstractEnsembleBackend
    threaded::Bool
end

CPUBackend(; threaded::Bool=false) = CPUBackend(threaded)

"""CUDA extension selector. Constructing it does not load CUDA.jl."""
struct CUDABackend <: AbstractEnsembleBackend end

"""Metal extension selector. Constructing it does not load Metal.jl."""
struct MetalBackend <: AbstractEnsembleBackend end

"""Error raised when an optional backend cannot prepare an ensemble."""
struct BackendUnavailableError <: Exception
    backend::Symbol
    reason::String
end

function Base.showerror(io::IO, error::BackendUnavailableError)
    print(io, error.backend, " ensemble backend is unavailable: ", error.reason)
end

"""Return the readiness state of an ensemble backend without running a solve."""
backend_status(::CPUBackend) = :available

function _extension_module(name::Symbol)
    VERSION < v"1.9" && return nothing
    return Base.get_extension(@__MODULE__, name)
end

function backend_status(::CUDABackend)
    extension_module = _extension_module(:ArrheniusCUDAExt)
    return isnothing(extension_module) ?
        :extension_not_loaded : extension_module.backend_status()
end

function backend_status(::MetalBackend)
    extension_module = _extension_module(:ArrheniusMetalExt)
    return isnothing(extension_module) ?
        :extension_not_loaded : extension_module.backend_status()
end

"""Prepared CPU ensemble returned by `prepare_ensemble`."""
struct PreparedEnsemble{B<:CPUBackend,M,R,P,F}
    backend::B
    mechanism::M
    reactor::R
    policies::P
    sample_solver::F
end

function _validate_manifest(mechanism::MechanismIR, manifest::EnsembleManifest)
    for sample in manifest.samples
        indices = sample.perturbations.reaction_indices
        if !isempty(indices) && last(indices) > mechanism.n_reactions
            throw(ArgumentError(
                "sample $(sample.id) perturbs reaction $(last(indices)), but " *
                "the mechanism has $(mechanism.n_reactions) reactions",
            ))
        end
    end
    return nothing
end

function _prepare_backend(
    backend::CPUBackend,
    mechanism::MechanismIR,
    reactor,
    policies::EnsemblePolicies,
    sample_solver,
)
    isnothing(sample_solver) && throw(ArgumentError(
        "the CPU backend requires " *
        "sample_solver(ir, reactor, sample, policies)",
    ))
    return PreparedEnsemble(
        backend,
        mechanism,
        reactor,
        policies,
        sample_solver,
    )
end

function _prepare_backend(
    backend::CUDABackend,
    mechanism::MechanismIR,
    reactor,
    policies::EnsemblePolicies,
    sample_solver,
)
    extension_module = _extension_module(:ArrheniusCUDAExt)
    isnothing(extension_module) && throw(BackendUnavailableError(
        :cuda,
        "load CUDA.jl to activate ArrheniusCUDAExt",
    ))
    return extension_module.prepare_ensemble(
        backend,
        mechanism,
        reactor,
        policies,
        sample_solver,
    )
end

function _prepare_backend(
    backend::MetalBackend,
    mechanism::MechanismIR,
    reactor,
    policies::EnsemblePolicies,
    sample_solver,
)
    extension_module = _extension_module(:ArrheniusMetalExt)
    isnothing(extension_module) && throw(BackendUnavailableError(
        :metal,
        "load Metal.jl to activate ArrheniusMetalExt",
    ))
    return extension_module.prepare_ensemble(
        backend,
        mechanism,
        reactor,
        policies,
        sample_solver,
    )
end

"""
    prepare_ensemble(mechanism, reactor; backend=CPUBackend(),
                     precision_policy=PrecisionPolicy(),
                     jacobian=FiniteDifferenceJacobian(),
                     linear_solver=DenseLinearSolver(), sample_solver=nothing)

Prepare mechanism- and reactor-specific resources independently of an ensemble
manifest. On CPU, `sample_solver` must implement
`sample_solver(ir, reactor, sample, policies)` and return a payload. The reactor
may be any concrete model or configuration object. Optional CUDA and Metal
packages are loaded only when requested; their solver implementations remain
extension responsibilities.
"""
function prepare_ensemble(
    mechanism,
    reactor;
    backend::AbstractEnsembleBackend=CPUBackend(),
    precision_policy::PrecisionPolicy=PrecisionPolicy(),
    jacobian::AbstractJacobianPolicy=FiniteDifferenceJacobian(),
    linear_solver::AbstractLinearSolverPolicy=DenseLinearSolver(),
    sample_solver=nothing,
)
    mechanism_ir = MechanismIR(mechanism)
    policies = EnsemblePolicies(precision_policy, jacobian, linear_solver)
    return _prepare_backend(
        backend,
        mechanism_ir,
        reactor,
        policies,
        sample_solver,
    )
end

function _evaluate_qoi(schema::QoISchema, payload, sample)
    if isempty(schema.reducers)
        payload isa NamedTuple || throw(ArgumentError(
            "a sample solver must return a NamedTuple when the QoI schema is empty",
        ))
        return payload
    end
    reduced = map(
        reducer -> reducer(payload, sample),
        values(schema.reducers),
    )
    return NamedTuple{keys(schema.reducers)}(reduced)
end

function _solve_one(
    prepared::PreparedEnsemble,
    qoi::QoISchema,
    sample,
)
    payload = prepared.sample_solver(
        prepared.mechanism,
        prepared.reactor,
        sample,
        prepared.policies,
    )
    return QoIResult(sample.id, _evaluate_qoi(qoi, payload, sample))
end

function _effective_batch_size(batch_policy, sample_count::Integer)
    if batch_policy === :auto
        return max(Int(sample_count), 1)
    elseif batch_policy isa Integer && !(batch_policy isa Bool) &&
           batch_policy > 0
        return Int(min(batch_policy, max(Int(sample_count), 1)))
    end
    throw(ArgumentError("batch_policy must be :auto or a positive integer"))
end

function _solve_indices!(
    destination,
    prepared::PreparedEnsemble,
    manifest::EnsembleManifest,
    first_index::Integer,
    batch_size::Integer,
)
    samples = manifest.samples
    first_index > length(samples) && return destination
    for batch_start in first_index:batch_size:length(samples)
        batch_stop = min(batch_start + batch_size - 1, length(samples))
        if prepared.backend.threaded && batch_stop > batch_start
            Threads.@threads for index in batch_start:batch_stop
                @inbounds destination[index] =
                    _solve_one(prepared, manifest.qoi, samples[index])
            end
        else
            @inbounds for index in batch_start:batch_stop
                destination[index] =
                    _solve_one(prepared, manifest.qoi, samples[index])
            end
        end
    end
    return destination
end

"""
    solve_ensemble!(destination, prepared, manifest; batch_policy=:auto)

Run a prepared CPU ensemble and replace every element of `destination` with a
`QoIResult`. Results preserve manifest order. `batch_policy` accepts `:auto` or
a positive integer. On CPU, an integer partitions host iteration into batches;
it does not imply accelerator-style kernel batching. With
`CPUBackend(threaded=true)`, the sample solver and its captured state must be
thread safe.
"""
function solve_ensemble!(
    destination::AbstractVector,
    prepared::PreparedEnsemble,
    manifest::EnsembleManifest;
    batch_policy=:auto,
)
    _validate_manifest(prepared.mechanism, manifest)
    samples = manifest.samples
    length(destination) == length(samples) || throw(DimensionMismatch(
        "destination must contain one entry per ensemble sample",
    ))
    batch_size = _effective_batch_size(batch_policy, length(samples))
    return _solve_indices!(destination, prepared, manifest, 1, batch_size)
end

function _solve_ensemble(
    prepared::PreparedEnsemble,
    manifest::EnsembleManifest,
    batch_policy,
)
    _validate_manifest(prepared.mechanism, manifest)
    samples = manifest.samples
    batch_size = _effective_batch_size(batch_policy, length(samples))
    isempty(samples) && return QoIResult[]
    first_result = _solve_one(prepared, manifest.qoi, first(samples))
    destination = Vector{typeof(first_result)}(undef, length(samples))
    destination[1] = first_result
    return _solve_indices!(
        destination,
        prepared,
        manifest,
        2,
        batch_size,
    )
end

"""End-to-end wall-clock profile returned by `profile_ensemble`."""
struct EnsembleProfile{B<:AbstractEnsembleBackend,R,P}
    backend::B
    sample_count::Int
    repetitions::Int
    warmup::Bool
    batch_policy::P
    effective_batch_size::Int
    elapsed_seconds::Vector{Float64}
    median_seconds::Float64
    samples_per_second::Float64
    results::R
end

function _median(values::AbstractVector)
    ordered = sort(values)
    middle = length(ordered) ÷ 2
    return isodd(length(ordered)) ?
        ordered[middle + 1] : (ordered[middle] + ordered[middle + 1]) / 2
end

"""
    profile_ensemble(prepared, manifest; repetitions=3, warmup=true,
                     batch_policy=:auto)

Measure end-to-end CPU ensemble time, including result allocation and QoI
reduction. If `warmup` is true, one unmeasured ensemble run occurs first. The
returned throughput uses the median measured time. `effective_batch_size`
records how the CPU host loop interpreted `batch_policy`; it is not a claim
about accelerator kernel batching.
"""
function profile_ensemble(
    prepared::PreparedEnsemble,
    manifest::EnsembleManifest;
    repetitions::Integer=3,
    warmup::Bool=true,
    batch_policy=:auto,
)
    repetitions > 0 || throw(ArgumentError("repetitions must be positive"))
    sample_count = length(manifest.samples)
    effective_batch_size = _effective_batch_size(batch_policy, sample_count)
    warmup && _solve_ensemble(prepared, manifest, batch_policy)
    elapsed_seconds = Vector{Float64}(undef, repetitions)
    results = nothing
    for repetition in 1:repetitions
        start_time = time_ns()
        results = _solve_ensemble(prepared, manifest, batch_policy)
        elapsed_seconds[repetition] = (time_ns() - start_time) / 1.0e9
    end
    median_seconds = _median(elapsed_seconds)
    throughput = median_seconds == 0.0 ? Inf : sample_count / median_seconds
    return EnsembleProfile(
        prepared.backend,
        sample_count,
        Int(repetitions),
        warmup,
        batch_policy,
        isempty(manifest.samples) ? 0 : effective_batch_size,
        elapsed_seconds,
        median_seconds,
        throughput,
        results,
    )
end

export MechanismIR
export SparseLogAPerturbations, materialize_rate_multipliers
export EnsembleSample, EnsembleManifest, QoISchema, QoIResult
export PrecisionPolicy, FiniteDifferenceJacobian, ProvidedJacobian
export DenseLinearSolver, SparseLinearSolver, ProvidedLinearSolver
export EnsemblePolicies
export CPUBackend, CUDABackend, MetalBackend, BackendUnavailableError
export PreparedEnsemble, EnsembleProfile, backend_status
export prepare_ensemble, solve_ensemble!, profile_ensemble

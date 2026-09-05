"""Reusable storage for exact reaction-rate partial derivatives.

The first implementation covers elementary, third-body, Lindemann, and Troe
rates. PLOG is rejected explicitly until its pressure interpolation derivative
is implemented. The derivatives are with respect to species concentrations at
fixed temperature and temperature at fixed concentrations.
At zero effective collider concentration, collider derivatives use the finite
right-hand limit from nonnegative concentrations.
"""
mutable struct KineticsDerivativeWorkspace{T<:AbstractFloat}
    effective_rates::Vector{T}
    forward_mass_action::Vector{T}
    reverse_mass_action::Vector{T}
    equilibrium_constants::Vector{T}
    delta_entropy::Vector{T}
    delta_enthalpy::Vector{T}
    dlog_rate_dT::Vector{T}
    dlog_rate_dlogM::Vector{T}
    collider_concentrations::Vector{T}
    falloff_map::Vector{Int32}
    three_body::BitVector
end

function KineticsDerivativeWorkspace(
    reaction::Reaction,
    ::Type{T}=Float64,
) where {T<:AbstractFloat}
    n = reaction.n_reactions
    falloff_map = zeros(Int32, n)
    @inbounds for (falloff_index, reaction_index) in enumerate(reaction.index_falloff)
        falloff_map[reaction_index] = Int32(falloff_index)
    end
    three_body = falses(n)
    three_body[reaction.index_three_body] .= true
    return KineticsDerivativeWorkspace{T}(
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
        falloff_map,
        three_body,
    )
end

export KineticsDerivativeWorkspace

"Return whether the exact rate-partial implementation covers every reaction."
supports_exact_rate_partials(reaction::Reaction) =
    isempty(reaction.plog.reaction_indices)

export supports_exact_rate_partials

@inline function _arrhenius_rate_and_log_derivative(coefficients, row, T)
    @inbounds A = coefficients[row, 1]
    @inbounds b = coefficients[row, 2]
    @inbounds Ea = coefficients[row, 3]
    inverse_temperature = inv(T)
    activation = oftype(T, 4184.0 / R) * inverse_temperature
    rate = A * exp(b * log(T) - Ea * activation)
    dlog_rate_dT = b * inverse_temperature +
        Ea * oftype(T, 4184.0 / R) * inverse_temperature^2
    return rate, dlog_rate_dT
end

@inline function _mass_action(concentrations, indices, orders, reaction_index)
    result = one(eltype(concentrations))
    @inbounds for species in indices[reaction_index]
        result *= concentrations[species]^orders[species, reaction_index]
    end
    return result
end

@inline function _mass_action_partial(
    concentrations,
    indices,
    orders,
    reaction_index,
    target_species,
)
    order = @inbounds orders[target_species, reaction_index]
    concentration = @inbounds concentrations[target_species]
    if iszero(concentration) && order < one(order)
        throw(DomainError(
            concentration,
            "a sub-unity or negative reaction order has a singular derivative at zero concentration",
        ))
    end
    result = order * concentration^(order - one(order))
    @inbounds for species in indices[reaction_index]
        species == target_species && continue
        result *= concentrations[species]^orders[species, reaction_index]
    end
    return result
end

@inline function _troe_factor_and_partials(T, reduced_pressure, parameters)
    @inbounds alpha = parameters[1]
    @inbounds T1 = parameters[2]
    @inbounds T2 = parameters[3]
    @inbounds T3 = parameters[4]
    term3 = (one(T) - alpha) * exp(-T / T3)
    term1 = alpha * exp(-T / T1)
    term2 = exp(-T2 / T)
    center = term3 + term1 + term2
    dcenter_dT = -term3 / T3 - term1 / T1 + term2 * T2 / T^2
    dlog_center_dT = dcenter_dT / center

    log_reduced = log10(reduced_pressure)
    log_center = log10(center)
    c_troe = -oftype(T, 0.4) - oftype(T, 0.67) * log_center
    n_troe = oftype(T, 0.75) - oftype(T, 1.27) * log_center
    numerator = log_reduced + c_troe
    denominator = n_troe - oftype(T, 0.14) * numerator
    f1 = numerator / denominator
    inverse_shape = inv(one(T) + f1^2)
    log_factor = log_center * inverse_shape
    factor = exp(log(oftype(T, 10.0)) * log_factor)

    # For either variable z, d(log(F))/dz = g_x*d(log(Pr))/dz +
    # g_L*d(log(Fcent))/dz. Factors of log(10) cancel because x and L are
    # base-10 logarithms while the returned inputs are natural logarithms.
    dnumerator_dx = one(T)
    ddenominator_dx = -oftype(T, 0.14)
    df1_dx = (
        dnumerator_dx * denominator - numerator * ddenominator_dx
    ) / denominator^2
    dlog_factor_dx = -log_center * oftype(T, 2.0) * f1 * df1_dx *
        inverse_shape^2

    dnumerator_dL = -oftype(T, 0.67)
    ddenominator_dL = -oftype(T, 1.27) -
        oftype(T, 0.14) * dnumerator_dL
    df1_dL = (
        dnumerator_dL * denominator - numerator * ddenominator_dL
    ) / denominator^2
    dlog_factor_dL = inverse_shape -
        log_center * oftype(T, 2.0) * f1 * df1_dL * inverse_shape^2
    return factor, dlog_factor_dx, dlog_factor_dL, dlog_center_dT
end

function _validate_rate_partial_storage(
    qdot,
    dqdot_dC,
    dqdot_dT,
    reaction,
    concentrations,
    workspace,
)
    n_reactions = reaction.n_reactions
    n_species = size(reaction.vk, 1)
    length(qdot) == n_reactions || throw(DimensionMismatch(
        "qdot must contain one entry per reaction",
    ))
    size(dqdot_dC) == (n_reactions, n_species) || throw(DimensionMismatch(
        "dqdot_dC must have shape n_reactions by n_species",
    ))
    length(dqdot_dT) == n_reactions || throw(DimensionMismatch(
        "dqdot_dT must contain one entry per reaction",
    ))
    length(concentrations) == n_species || throw(DimensionMismatch(
        "concentrations must contain one entry per species",
    ))
    length(workspace.effective_rates) == n_reactions || throw(DimensionMismatch(
        "derivative workspace does not match the reaction count",
    ))
    return nothing
end

"""
    reaction_rate_partials!(qdot, dqdot_dC, dqdot_dT, reaction, T, C,
                            S0, h_mole, workspace; rate_multipliers=nothing)

Evaluate net rates of progress and their exact partial derivatives. `dqdot_dC`
is ordered reaction by species, and `dqdot_dT` differentiates at fixed
concentrations. `S0` and `h_mole` are the standard-state entropy and molar
enthalpy arrays used by `wdot!`.

The method preserves the complete mechanism. It performs no QSSA, mechanism
reduction, tabulation, or rate approximation.
"""
function reaction_rate_partials!(
    qdot::AbstractVector{T},
    dqdot_dC::AbstractMatrix{T},
    dqdot_dT::AbstractVector{T},
    reaction::Reaction,
    temperature::T,
    concentrations::AbstractVector{T},
    entropies::AbstractVector{T},
    enthalpies::AbstractVector{T},
    workspace::KineticsDerivativeWorkspace{T};
    rate_multipliers=nothing,
) where {T<:AbstractFloat}
    supports_exact_rate_partials(reaction) || throw(ArgumentError(
        "exact rate partials do not yet support PLOG reactions",
    ))
    _validate_rate_partial_storage(
        qdot,
        dqdot_dC,
        dqdot_dT,
        reaction,
        concentrations,
        workspace,
    )
    length(entropies) == length(concentrations) || throw(DimensionMismatch(
        "entropies must contain one entry per species",
    ))
    length(enthalpies) == length(concentrations) || throw(DimensionMismatch(
        "enthalpies must contain one entry per species",
    ))
    if !isnothing(rate_multipliers)
        length(rate_multipliers) == reaction.n_reactions ||
            throw(DimensionMismatch(
                "rate_multipliers must contain one value per reaction",
            ))
    end
    temperature > zero(T) || throw(DomainError(
        temperature,
        "temperature must be positive",
    ))
    all(>=(zero(T)), concentrations) || throw(DomainError(
        minimum(concentrations),
        "concentrations must be nonnegative",
    ))

    fill!(dqdot_dC, zero(T))
    mul!(workspace.delta_entropy, transpose(reaction.vk), entropies)
    mul!(workspace.delta_enthalpy, transpose(reaction.vk), enthalpies)
    efficiency_rows = rowvals(reaction.efficiencies_coeffs)
    efficiency_values = nonzeros(reaction.efficiencies_coeffs)
    gas_constant = oftype(temperature, R)
    one_atmosphere = oftype(temperature, one_atm)

    @inbounds for reaction_index in 1:reaction.n_reactions
        effective_rate, dlog_rate_dT = _arrhenius_rate_and_log_derivative(
            reaction.Arrhenius_coeffs,
            reaction_index,
            temperature,
        )
        dlog_rate_dlogM = zero(T)
        zero_collider_slope = zero(T)
        collider = zero(T)
        falloff_index = workspace.falloff_map[reaction_index]
        if falloff_index > 0 || workspace.three_body[reaction_index]
            for pointer in nzrange(reaction.efficiencies_coeffs, reaction_index)
                collider += efficiency_values[pointer] *
                    concentrations[efficiency_rows[pointer]]
            end
            if collider <= zero(T)
                if iszero(collider)
                    zero_collider_slope = effective_rate
                    if falloff_index > 0
                        low_rate, _ = _arrhenius_rate_and_log_derivative(
                            reaction.Arrhenius_0, falloff_index, temperature,
                        )
                        zero_collider_slope = _zero_pressure_falloff(
                            low_rate, one(T), temperature, reaction, falloff_index,
                        )
                    end
                end
                effective_rate = zero(T)
                dlog_rate_dT = zero(T)
                dlog_rate_dlogM = zero(T)
            elseif falloff_index > 0
                low_rate, dlog_low_dT = _arrhenius_rate_and_log_derivative(
                    reaction.Arrhenius_0,
                    falloff_index,
                    temperature,
                )
                reduced_pressure = low_rate * collider / effective_rate
                inverse_one_plus_reduced = inv(one(T) + reduced_pressure)
                effective_rate *= reduced_pressure * inverse_one_plus_reduced
                dlog_reduced_dT = dlog_low_dT - dlog_rate_dT
                dlog_rate_dlogM = inverse_one_plus_reduced
                dlog_rate_dT += inverse_one_plus_reduced * dlog_reduced_dT
                troe_index = reaction.index_falloff_Troe[falloff_index]
                if troe_index > 0
                    factor, dlogF_dlogPr, dlogF_dlogFcent,
                    dlogFcent_dT = _troe_factor_and_partials(
                        temperature,
                        reduced_pressure,
                        @view(reaction.Troe_[troe_index, :]),
                    )
                    effective_rate *= factor
                    dlog_rate_dlogM += dlogF_dlogPr
                    dlog_rate_dT += dlogF_dlogPr * dlog_reduced_dT +
                        dlogF_dlogFcent * dlogFcent_dT
                end
            else
                effective_rate *= collider
                dlog_rate_dlogM = one(T)
            end
        end
        if !isnothing(rate_multipliers)
            multiplier = rate_multipliers[reaction_index]
            isfinite(multiplier) && multiplier > zero(multiplier) ||
                throw(ArgumentError(
                    "rate multipliers must be finite and positive",
                ))
            effective_rate *= multiplier
            zero_collider_slope *= multiplier
        end
        workspace.effective_rates[reaction_index] = effective_rate
        workspace.dlog_rate_dT[reaction_index] = dlog_rate_dT
        workspace.dlog_rate_dlogM[reaction_index] = dlog_rate_dlogM
        workspace.collider_concentrations[reaction_index] = collider

        forward_mass_action = _mass_action(
            concentrations,
            reaction.i_reactant,
            reaction.reactant_orders,
            reaction_index,
        )
        reverse_mass_action = reaction.is_reversible[reaction_index] ?
            _mass_action(
                concentrations,
                reaction.i_product,
                reaction.product_stoich_coeffs,
                reaction_index,
            ) : zero(T)
        log_equilibrium =
            workspace.delta_entropy[reaction_index] / gas_constant -
            workspace.delta_enthalpy[reaction_index] /
                (gas_constant * temperature) +
            log(one_atmosphere / gas_constant / temperature) *
                reaction.vk_sum[reaction_index]
        equilibrium = exp(log_equilibrium)
        workspace.equilibrium_constants[reaction_index] = equilibrium
        workspace.forward_mass_action[reaction_index] = forward_mass_action
        workspace.reverse_mass_action[reaction_index] = reverse_mass_action
        reverse_scale = reaction.is_reversible[reaction_index] ?
            inv(equilibrium) : zero(T)
        forward_progress = effective_rate * forward_mass_action
        reverse_progress = effective_rate * reverse_scale * reverse_mass_action
        qdot[reaction_index] = forward_progress - reverse_progress

        dlog_equilibrium_dT =
            workspace.delta_enthalpy[reaction_index] /
                (gas_constant * temperature^2) -
            reaction.vk_sum[reaction_index] / temperature
        dqdot_dT[reaction_index] =
            qdot[reaction_index] * dlog_rate_dT +
            reverse_progress * dlog_equilibrium_dT

        for species in reaction.i_reactant[reaction_index]
            dqdot_dC[reaction_index, species] += effective_rate *
                _mass_action_partial(
                    concentrations,
                    reaction.i_reactant,
                    reaction.reactant_orders,
                    reaction_index,
                    species,
                )
        end
        if reaction.is_reversible[reaction_index]
            for species in reaction.i_product[reaction_index]
                dqdot_dC[reaction_index, species] -=
                    effective_rate * reverse_scale * _mass_action_partial(
                        concentrations,
                        reaction.i_product,
                        reaction.product_stoich_coeffs,
                        reaction_index,
                        species,
                    )
            end
        end
        if !iszero(zero_collider_slope) || !iszero(dlog_rate_dlogM)
            # The log derivative is singular at zero; use the finite right limit
            # of dk/d[M] before multiplying by the net mass-action factor.
            common = iszero(collider) ?
                zero_collider_slope * (forward_mass_action - reverse_scale * reverse_mass_action) :
                qdot[reaction_index] * dlog_rate_dlogM / collider
            for pointer in nzrange(reaction.efficiencies_coeffs, reaction_index)
                species = efficiency_rows[pointer]
                dqdot_dC[reaction_index, species] +=
                    common * efficiency_values[pointer]
            end
        end
    end
    return qdot, dqdot_dC, dqdot_dT
end

export reaction_rate_partials!

"Reusable arrays for an exact constant-volume mass-fraction Jacobian."
mutable struct ConstantVolumeJacobianWorkspace{T<:AbstractFloat}
    mole_fractions::Vector{T}
    concentrations::Vector{T}
    cp_R::Vector{T}
    dcp_R_dT::Vector{T}
    enthalpies::Vector{T}
    entropies::Vector{T}
    internal_energies::Vector{T}
    source::Vector{T}
    qdot::Vector{T}
    dqdot_dC::Matrix{T}
    dqdot_dT::Vector{T}
    dsource_dC::Matrix{T}
    dsource_dT::Vector{T}
    kinetics::KineticsDerivativeWorkspace{T}
end

function ConstantVolumeJacobianWorkspace(
    gas::Solution,
    ::Type{T}=Float64,
) where {T<:AbstractFloat}
    n_species = gas.n_species
    n_reactions = gas.n_reactions
    return ConstantVolumeJacobianWorkspace{T}(
        zeros(T, n_species),
        zeros(T, n_species),
        zeros(T, n_species),
        zeros(T, n_species),
        zeros(T, n_species),
        zeros(T, n_species),
        zeros(T, n_species),
        zeros(T, n_species),
        zeros(T, n_reactions),
        zeros(T, n_reactions, n_species),
        zeros(T, n_reactions),
        zeros(T, n_species, n_species),
        zeros(T, n_species),
        KineticsDerivativeWorkspace(gas.reaction, T),
    )
end

export ConstantVolumeJacobianWorkspace

@inline function _nasa_cp_R_temperature_derivative(
    thermo::IdealGasThermo,
    species,
    temperature,
)
    coefficients = _nasa7_coefficients(thermo, species, temperature)
    temperature2 = temperature^2
    temperature3 = temperature2 * temperature
    @inbounds return (
        coefficients[species, 2] +
        oftype(temperature, 2.0) * coefficients[species, 3] * temperature +
        oftype(temperature, 3.0) * coefficients[species, 4] * temperature2 +
        oftype(temperature, 4.0) * coefficients[species, 5] * temperature3
    )
end

function _validate_constant_volume_jacobian_storage(
    jacobian,
    rhs,
    gas,
    state,
    workspace,
)
    dimension = gas.n_species + 1
    size(jacobian) == (dimension, dimension) || throw(DimensionMismatch(
        "jacobian must have shape (n_species + 1, n_species + 1)",
    ))
    length(rhs) == dimension || throw(DimensionMismatch(
        "rhs must contain mass-fraction derivatives followed by temperature",
    ))
    length(state) == dimension || throw(DimensionMismatch(
        "state must contain mass fractions followed by temperature",
    ))
    length(workspace.source) == gas.n_species || throw(DimensionMismatch(
        "constant-volume Jacobian workspace does not match the mechanism",
    ))
    return nothing
end

"""
    constant_volume_jacobian!(jacobian, rhs, gas, density, state, workspace;
                              rate_multipliers=nothing)

Evaluate the exact RHS and dense state Jacobian for a homogeneous, adiabatic,
closed constant-volume ideal-gas reactor. The state is ordered as mass
fractions followed by temperature. Density is held fixed, so concentration is
`density * Y_k / molecular_weight_k` and pressure follows the ideal-gas law.

This first generated-formula implementation supports the reaction families
reported by `supports_exact_rate_partials`. It is intended both as a numerical
oracle for accelerator kernels and as a way to eliminate finite-difference RHS
calls; sparse backends may scatter only their fixed structural entries.
"""
function constant_volume_jacobian!(
    jacobian::AbstractMatrix{T},
    rhs::AbstractVector{T},
    gas::Solution,
    density::T,
    state::AbstractVector{T},
    workspace::ConstantVolumeJacobianWorkspace{T};
    rate_multipliers=nothing,
) where {T<:AbstractFloat}
    _validate_constant_volume_jacobian_storage(
        jacobian,
        rhs,
        gas,
        state,
        workspace,
    )
    density > zero(T) || throw(DomainError(density, "density must be positive"))
    n_species = gas.n_species
    mass_fractions = @view state[1:n_species]
    temperature = state[end]
    temperature > zero(T) || throw(DomainError(
        temperature,
        "temperature must be positive",
    ))
    all(>=(zero(T)), mass_fractions) || throw(DomainError(
        minimum(mass_fractions),
        "mass fractions must be nonnegative",
    ))

    inverse_mean_molecular_weight = zero(T)
    @inbounds for species in 1:n_species
        weighted = mass_fractions[species] / gas.MW[species]
        inverse_mean_molecular_weight += weighted
        workspace.concentrations[species] = density * weighted
    end
    mean_molecular_weight = inv(inverse_mean_molecular_weight)
    @inbounds for species in 1:n_species
        workspace.mole_fractions[species] =
            mass_fractions[species] * mean_molecular_weight / gas.MW[species]
    end
    pressure = density * T(R) * temperature *
        inverse_mean_molecular_weight
    cal_cp_R!(
        workspace.cp_R,
        gas,
        temperature,
        pressure,
        workspace.mole_fractions,
    )
    cal_h_RT!(
        workspace.enthalpies,
        gas,
        temperature,
        pressure,
        workspace.mole_fractions,
    )
    cal_s0_R!(
        workspace.entropies,
        gas,
        temperature,
        pressure,
        workspace.mole_fractions,
    )
    gas_constant = T(R)
    heat_capacity_volume_mass = zero(T)
    dcv_dT = zero(T)
    @inbounds for species in 1:n_species
        cp_R = workspace.cp_R[species]
        dcp_R_dT = _nasa_cp_R_temperature_derivative(
            gas.thermo,
            species,
            temperature,
        )
        workspace.dcp_R_dT[species] = dcp_R_dT
        workspace.enthalpies[species] *= gas_constant * temperature
        workspace.entropies[species] *= gas_constant
        workspace.internal_energies[species] =
            workspace.enthalpies[species] - gas_constant * temperature
        weighted_mass_fraction = mass_fractions[species] / gas.MW[species]
        heat_capacity_volume_mass +=
            gas_constant * weighted_mass_fraction * (cp_R - one(T))
        dcv_dT += gas_constant * weighted_mass_fraction * dcp_R_dT
    end

    reaction_rate_partials!(
        workspace.qdot,
        workspace.dqdot_dC,
        workspace.dqdot_dT,
        gas.reaction,
        temperature,
        workspace.concentrations,
        workspace.entropies,
        workspace.enthalpies,
        workspace.kinetics;
        rate_multipliers=rate_multipliers,
    )
    mul!(workspace.source, gas.reaction.vk, workspace.qdot)
    mul!(workspace.dsource_dC, gas.reaction.vk, workspace.dqdot_dC)
    mul!(workspace.dsource_dT, gas.reaction.vk, workspace.dqdot_dT)

    internal_energy_source = dot(
        workspace.internal_energies,
        workspace.source,
    )
    inverse_density_cv = inv(density * heat_capacity_volume_mass)
    @inbounds for output_species in 1:n_species
        molecular_weight = gas.MW[output_species]
        rhs[output_species] =
            molecular_weight * workspace.source[output_species] / density
        for input_species in 1:n_species
            jacobian[output_species, input_species] =
                molecular_weight / gas.MW[input_species] *
                workspace.dsource_dC[output_species, input_species]
        end
        jacobian[output_species, end] = molecular_weight / density *
            workspace.dsource_dT[output_species]
    end
    rhs[end] = -internal_energy_source * inverse_density_cv

    @inbounds for input_species in 1:n_species
        dsource_dY_scale = density / gas.MW[input_species]
        dinternal_energy_source = zero(T)
        for output_species in 1:n_species
            dinternal_energy_source +=
                workspace.internal_energies[output_species] *
                workspace.dsource_dC[output_species, input_species] *
                dsource_dY_scale
        end
        dcv_dY = gas_constant / gas.MW[input_species] *
            (workspace.cp_R[input_species] - one(T))
        jacobian[end, input_species] =
            -dinternal_energy_source * inverse_density_cv +
            internal_energy_source * dcv_dY /
                (density * heat_capacity_volume_mass^2)
    end

    dinternal_energy_source_dT = zero(T)
    @inbounds for species in 1:n_species
        du_dT = gas_constant * (workspace.cp_R[species] - one(T))
        dinternal_energy_source_dT +=
            du_dT * workspace.source[species] +
            workspace.internal_energies[species] * workspace.dsource_dT[species]
    end
    jacobian[end, end] =
        -dinternal_energy_source_dT * inverse_density_cv +
        internal_energy_source * dcv_dT /
            (density * heat_capacity_volume_mass^2)
    return rhs, jacobian
end

export constant_volume_jacobian!

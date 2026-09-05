"Reusable storage for the dense exact constant-pressure reactor Jacobian."
struct ConstantPressureJacobianWorkspace{T<:AbstractFloat}
    chemistry::ConstantVolumeJacobianWorkspace{T}
end
ConstantPressureJacobianWorkspace(gas::Solution, ::Type{T}=Float64) where {T<:AbstractFloat} =
    ConstantPressureJacobianWorkspace(ConstantVolumeJacobianWorkspace(gas, T))

"""
    constant_pressure_jacobian!(jacobian, rhs, gas, pressure, state, workspace;
                               rate_multipliers=nothing, allow_signed_state=false)

Exact dense Jacobian and RHS for a closed adiabatic ideal-gas reactor at fixed
pressure, in independent mass fractions followed by temperature. Includes the
composition and temperature dependence of density. No species is eliminated.
The opt-in signed-state extension matches the existing RHS during Newton
iterations; it does not clamp mass fractions or make fractional powers at
negative concentrations well defined. Temperature and mean molecular weight
must remain positive.
"""
function constant_pressure_jacobian!(
    jacobian::AbstractMatrix{T}, rhs::AbstractVector{T}, gas::Solution,
    pressure::T, state::AbstractVector{T}, storage::ConstantPressureJacobianWorkspace{T};
    rate_multipliers=nothing,
    allow_signed_state=false,
) where {T<:AbstractFloat}
    work = storage.chemistry
    _validate_constant_volume_jacobian_storage(jacobian, rhs, gas, state, work)
    pressure > zero(T) || throw(DomainError(pressure, "pressure must be positive"))
    ns = gas.n_species
    Y = @view state[1:ns]
    temperature = state[end]
    temperature > zero(T) || throw(DomainError(temperature, "temperature must be positive"))
    (allow_signed_state || all(>=(zero(T)), Y)) || throw(DomainError(minimum(Y), "mass fractions must be nonnegative"))
    inverse_mean = zero(T)
    for i in 1:ns
        inverse_mean += Y[i] / gas.MW[i]
    end
    inverse_mean > zero(T) || throw(DomainError(inverse_mean, "composition must be nonzero"))
    mean_mw = inv(inverse_mean)
    gas_constant = T(R)
    density = pressure * mean_mw / (gas_constant * temperature)
    @inbounds for i in 1:ns
        work.mole_fractions[i] = Y[i] * mean_mw / gas.MW[i]
        work.concentrations[i] = density * Y[i] / gas.MW[i]
    end
    cal_cp_R!(work.cp_R, gas, temperature, pressure, work.mole_fractions)
    cal_h_RT!(work.enthalpies, gas, temperature, pressure, work.mole_fractions)
    cal_s0_R!(work.entropies, gas, temperature, pressure, work.mole_fractions)
    cp_mass = zero(T)
    dcp_dT = zero(T)
    @inbounds for i in 1:ns
        work.enthalpies[i] *= gas_constant * temperature
        work.entropies[i] *= gas_constant
        cp_mass += gas_constant * Y[i] / gas.MW[i] * work.cp_R[i]
        dcp_dT += gas_constant * Y[i] / gas.MW[i] *
            _nasa_cp_R_temperature_derivative(gas.thermo, i, temperature)
    end
    reaction_rate_partials!(work.qdot, work.dqdot_dC, work.dqdot_dT,
        gas.reaction, temperature, work.concentrations, work.entropies,
        work.enthalpies, work.kinetics; rate_multipliers=rate_multipliers,
        allow_signed_concentrations=allow_signed_state)
    mul!(work.source, gas.reaction.vk, work.qdot)
    mul!(work.dsource_dC, gas.reaction.vk, work.dqdot_dC)
    mul!(work.dsource_dT, gas.reaction.vk, work.dqdot_dT)
    # This existing vector is unused for internal energies in a pressure reactor.
    source_density_direction = work.internal_energies
    mul!(source_density_direction, work.dsource_dC, work.concentrations)
    heat_source = dot(work.enthalpies, work.source)
    inverse_density_cp = inv(density * cp_mass)
    rhs[end] = -heat_source * inverse_density_cp
    @inbounds for i in 1:ns
        rhs[i] = gas.MW[i] * work.source[i] / density
    end
    @inbounds for j in 1:ns
        concentration_scale = density / gas.MW[j]
        density_log_derivative = -mean_mw / gas.MW[j]
        heat_derivative = zero(T)
        for i in 1:ns
            source_derivative = work.dsource_dC[i, j] * concentration_scale +
                source_density_direction[i] * density_log_derivative
            jacobian[i, j] = gas.MW[i] / density *
                (source_derivative - work.source[i] * density_log_derivative)
            heat_derivative += work.enthalpies[i] * source_derivative
        end
        denominator_log_derivative = density_log_derivative +
            gas_constant * work.cp_R[j] / (gas.MW[j] * cp_mass)
        jacobian[end, j] = -heat_derivative * inverse_density_cp -
            rhs[end] * denominator_log_derivative
    end
    heat_temperature_derivative = zero(T)
    @inbounds for i in 1:ns
        source_derivative = work.dsource_dT[i] - source_density_direction[i] / temperature
        jacobian[i, end] = gas.MW[i] / density *
            (source_derivative + work.source[i] / temperature)
        heat_temperature_derivative += work.enthalpies[i] * source_derivative +
            gas_constant * work.cp_R[i] * work.source[i]
    end
    jacobian[end, end] = -heat_temperature_derivative * inverse_density_cp -
        rhs[end] * (dcp_dT / cp_mass - inv(temperature))
    return rhs, jacobian
end

export ConstantPressureJacobianWorkspace, constant_pressure_jacobian!

import ForwardDiff

# Branch on the physical value, not on the infinitesimal AD seed. ForwardDiff
# 1.x deliberately includes partials when comparing two equal primal values.
@inline _falloff_value(value) = value
@inline _falloff_value(value::ForwardDiff.Dual) = _falloff_value(ForwardDiff.value(value))

@inline function _zero_pressure_falloff(k0, collider, T, reaction, falloff_index)
    rate = k0 * collider
    troe_index = reaction.index_falloff_Troe[falloff_index]
    if troe_index > 0
        @inbounds F_cent =
            (one(T) - reaction.Troe_[troe_index, 1]) * exp(-T / reaction.Troe_[troe_index, 4]) +
            reaction.Troe_[troe_index, 1] * exp(-T / reaction.Troe_[troe_index, 2]) +
            exp(-reaction.Troe_[troe_index, 3] / T)
        # As log10(Pr) -> -Inf, the Troe ratio approaches -1 / 0.14.
        # Retaining collider (including its AD seed) preserves dk/d[M] at zero.
        exponent = inv(one(T) + inv(oftype(T, 0.14))^2)
        rate *= exp(log(F_cent) * exponent)
    end
    return rate
end

@inline function _plog_group_rate(plog::PlogData, group, T, logT)
    rate = zero(T)
    activation_scale = oftype(T, 4184.0 / R) / T
    @inbounds for row in plog.rate_offsets[group]:(plog.rate_offsets[group + 1] - 1)
        A = plog.Arrhenius_coeffs[row, 1]
        b = plog.Arrhenius_coeffs[row, 2]
        Ea = plog.Arrhenius_coeffs[row, 3]
        rate += A * exp(b * logT - Ea * activation_scale)
    end
    return rate
end

@inline function _plog_rate(plog::PlogData, plog_index, T, P, logT)
    first_group = plog.group_offsets[plog_index]
    last_group = plog.group_offsets[plog_index + 1] - 1

    if P <= plog.pressures[first_group]
        return _plog_group_rate(plog, first_group, T, logT)
    elseif P >= plog.pressures[last_group]
        return _plog_group_rate(plog, last_group, T, logT)
    end

    lower_group = first_group
    @inbounds while P > plog.pressures[lower_group + 1]
        lower_group += 1
    end
    upper_group = lower_group + 1
    lower_rate = _plog_group_rate(plog, lower_group, T, logT)
    upper_rate = _plog_group_rate(plog, upper_group, T, logT)
    @inbounds fraction =
        log(P / plog.pressures[lower_group]) /
        log(plog.pressures[upper_group] / plog.pressures[lower_group])
    # Cantera sums duplicate expressions at one pressure before taking the log.
    tiny = _positive_floor(lower_rate)
    return exp(log(lower_rate + tiny) +
               fraction * (log(upper_rate + tiny) - log(lower_rate + tiny)))
end

@inline _positive_floor(value::T) where {T<:AbstractFloat} = nextfloat(zero(T))
@inline _positive_floor(value) = oftype(value, 1.0e-300)

"""Log-domain Arrhenius parameters for mixed-precision rate evaluation.

The outer thermochemical state and source accumulation retain their existing
precision. Only the stored Arrhenius parameters and elementary exponential use
`T`, which is typically `Float32`.
"""
struct LogRateData{T<:AbstractFloat}
    base_log_a::Vector{T}
    base_b::Vector{T}
    base_ea::Vector{T}
    low_log_a::Vector{T}
    low_b::Vector{T}
    low_ea::Vector{T}
    plog_log_a::Vector{T}
    plog_b::Vector{T}
    plog_ea::Vector{T}
end

function LogRateData(reaction::Reaction, ::Type{T}=Float32) where {T<:AbstractFloat}
    arrays = (
        reaction.Arrhenius_coeffs,
        reaction.Arrhenius_0,
        reaction.plog.Arrhenius_coeffs,
    )
    all(array -> all(>(0), @view(array[:, 1])), arrays) ||
        throw(ArgumentError("log-domain rates require positive pre-exponentials"))
    convert_column(array, column) = T.(array[:, column])
    log_column(array) = T.(log.(@view array[:, 1]))
    return LogRateData(
        log_column(arrays[1]),
        convert_column(arrays[1], 2),
        convert_column(arrays[1], 3),
        log_column(arrays[2]),
        convert_column(arrays[2], 2),
        convert_column(arrays[2], 3),
        log_column(arrays[3]),
        convert_column(arrays[3], 2),
        convert_column(arrays[3], 3),
    )
end
export LogRateData

@inline function _mixed_elementary_rate(log_a, b, ea, logT, activation_scale, outer)
    return oftype(outer, exp(log_a + b * logT - ea * activation_scale))
end

@inline function _plog_group_rate(
    rate_data::LogRateData,
    plog::PlogData,
    group,
    outer_temperature,
    rate_logT,
    rate_activation_scale,
)
    rate = zero(outer_temperature)
    @inbounds for row in plog.rate_offsets[group]:(plog.rate_offsets[group + 1] - 1)
        rate += _mixed_elementary_rate(
            rate_data.plog_log_a[row],
            rate_data.plog_b[row],
            rate_data.plog_ea[row],
            rate_logT,
            rate_activation_scale,
            outer_temperature,
        )
    end
    return rate
end

@inline function _plog_rate(
    rate_data::LogRateData,
    plog::PlogData,
    plog_index,
    T,
    P,
    rate_logT,
    rate_activation_scale,
)
    first_group = plog.group_offsets[plog_index]
    last_group = plog.group_offsets[plog_index + 1] - 1
    if P <= plog.pressures[first_group]
        return _plog_group_rate(
            rate_data,
            plog,
            first_group,
            T,
            rate_logT,
            rate_activation_scale,
        )
    elseif P >= plog.pressures[last_group]
        return _plog_group_rate(
            rate_data,
            plog,
            last_group,
            T,
            rate_logT,
            rate_activation_scale,
        )
    end
    lower_group = first_group
    @inbounds while P > plog.pressures[lower_group + 1]
        lower_group += 1
    end
    upper_group = lower_group + 1
    lower_rate = _plog_group_rate(
        rate_data,
        plog,
        lower_group,
        T,
        rate_logT,
        rate_activation_scale,
    )
    upper_rate = _plog_group_rate(
        rate_data,
        plog,
        upper_group,
        T,
        rate_logT,
        rate_activation_scale,
    )
    @inbounds fraction =
        log(P / plog.pressures[lower_group]) /
        log(plog.pressures[upper_group] / plog.pressures[lower_group])
    tiny = _positive_floor(lower_rate)
    return exp(
        log(lower_rate + tiny) +
        fraction * (log(upper_rate + tiny) - log(lower_rate + tiny)),
    )
end

@inline function _plog_rate_with_collider(
    rate_data::LogRateData,
    plog::PlogData,
    plog_index,
    T,
    P,
    rate_logT,
    rate_activation_scale,
    C,
)
    rate = _plog_rate(
        rate_data,
        plog,
        plog_index,
        T,
        P,
        rate_logT,
        rate_activation_scale,
    )
    @inbounds collider_index = plog.collider_indices[plog_index]
    return collider_index > 0 ? rate * C[collider_index] : rate
end

@inline function _plog_rate_with_collider(
    plog::PlogData,
    plog_index,
    T,
    P,
    logT,
    C,
)
    rate = _plog_rate(plog, plog_index, T, P, logT)
    @inbounds collider_index = plog.collider_indices[plog_index]
    return collider_index > 0 ? rate * C[collider_index] : rate
end

"Reusable arrays for allocation-free kinetics evaluation."
mutable struct KineticsWorkspace{T}
    kf::Vector{T}
    kr::Vector{T}
    delta_s::Vector{T}
    delta_h::Vector{T}
    equilibrium_constants::Vector{T}
    rates_of_progress::Vector{T}
end

function KineticsWorkspace(reaction::Reaction, ::Type{T}=Float64) where {T}
    n = reaction.n_reactions
    return KineticsWorkspace{T}(
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
    )
end
export KineticsWorkspace

"Compute reaction source terms into preallocated storage."
function wdot!(
    wdot,
    reaction,
    T,
    C,
    S0,
    h_mole,
    workspace;
    get_qdot=false,
    rate_multipliers=nothing,
    log_rate_data=nothing,
)
    kf = workspace.kf
    kr = workspace.kr
    logT = log(T)
    gas_constant = oftype(T, R)
    one_atmosphere = oftype(T, one_atm)
    activation_scale = oftype(T, 4184.0 / R) / T
    if !isnothing(log_rate_data)
        length(log_rate_data.base_log_a) == length(kf) ||
            throw(DimensionMismatch(
                "log_rate_data must contain one base rate per reaction",
            ))
        RateScalar = eltype(log_rate_data.base_log_a)
        rate_temperature = RateScalar(T)
        rate_logT = log(rate_temperature)
        rate_activation_scale = RateScalar(4184.0 / R) / rate_temperature
    end
    @inbounds for i in eachindex(kf)
        if isnothing(log_rate_data)
            kf[i] = reaction.Arrhenius_coeffs[i, 1] * exp(
                reaction.Arrhenius_coeffs[i, 2] * logT -
                reaction.Arrhenius_coeffs[i, 3] * activation_scale,
            )
        else
            kf[i] = _mixed_elementary_rate(
                log_rate_data.base_log_a[i],
                log_rate_data.base_b[i],
                log_rate_data.base_ea[i],
                rate_logT,
                rate_activation_scale,
                T,
            )
        end
    end

    if !isempty(reaction.plog.reaction_indices)
        P = sum(C) * R * T
        for (plog_index, reaction_index) in enumerate(reaction.plog.reaction_indices)
            @inbounds kf[reaction_index] = if isnothing(log_rate_data)
                _plog_rate_with_collider(
                    reaction.plog,
                    plog_index,
                    T,
                    P,
                    logT,
                    C,
                )
            else
                _plog_rate_with_collider(
                    log_rate_data,
                    reaction.plog,
                    plog_index,
                    T,
                    P,
                    rate_logT,
                    rate_activation_scale,
                    C,
                )
            end
        end
    end

    for i in reaction.index_three_body
        @inbounds kf[i] *= dot(@view(reaction.efficiencies_coeffs[:, i]), C)
    end

    for (j, i) in enumerate(reaction.index_falloff)
        @inbounds A0 = reaction.Arrhenius_0[j, 1]
        @inbounds b0 = reaction.Arrhenius_0[j, 2]
        @inbounds Ea0 = reaction.Arrhenius_0[j, 3]
        k0 = if isnothing(log_rate_data)
            A0 * exp(b0 * logT - Ea0 * activation_scale)
        else
            _mixed_elementary_rate(
                log_rate_data.low_log_a[j],
                log_rate_data.low_b[j],
                log_rate_data.low_ea[j],
                rate_logT,
                rate_activation_scale,
                T,
            )
        end
        @inbounds collider = dot(@view(reaction.efficiencies_coeffs[:, i]), C)
        if _falloff_value(collider) < zero(_falloff_value(collider))
            kf[i] = zero(collider)
            continue
        elseif iszero(_falloff_value(collider))
            kf[i] = _zero_pressure_falloff(k0, collider, T, reaction, j)
            continue
        end
        @inbounds Pr = k0 * collider / kf[i]
        if _falloff_value(Pr) <= zero(_falloff_value(Pr))
            kf[i] = _zero_pressure_falloff(k0, collider, T, reaction, j)
            continue
        end
        lPr = log10(Pr)
        @inbounds kf[i] *= Pr / (1 + Pr)

        if reaction.index_falloff_Troe[j] > 0
            k = reaction.index_falloff_Troe[j]
            @inbounds F_cent =
                (one(T) - reaction.Troe_[k, 1]) * exp(-T / reaction.Troe_[k, 4]) +
                reaction.Troe_[k, 1] * exp(-T / reaction.Troe_[k, 2]) +
                exp(-reaction.Troe_[k, 3] / T)
            lF_cent = log10(F_cent)
            C_troe = -oftype(T, 0.4) - oftype(T, 0.67) * lF_cent
            N = oftype(T, 0.75) - oftype(T, 1.27) * lF_cent
            f1 = (lPr + C_troe) /
                (N - oftype(T, 0.14) * (lPr + C_troe))
            @inbounds kf[i] *= exp(
                log(oftype(T, 10.0)) * lF_cent / (one(T) + f1^2),
            )
        end
    end

    if !isnothing(rate_multipliers)
        length(rate_multipliers) == length(kf) || throw(DimensionMismatch(
            "rate_multipliers must contain one value per reaction",
        ))
        @inbounds for i in eachindex(kf)
            kf[i] *= rate_multipliers[i]
        end
    end

    mul!(workspace.delta_s, transpose(reaction.vk), S0)
    mul!(workspace.delta_h, transpose(reaction.vk), h_mole)
    @inbounds for i in eachindex(kf)
        workspace.equilibrium_constants[i] = exp(
            workspace.delta_s[i] / gas_constant -
            workspace.delta_h[i] / (gas_constant * T) +
            log(one_atmosphere / gas_constant / T) * reaction.vk_sum[i],
        )
        kr[i] = reaction.is_reversible[i] ?
            kf[i] / workspace.equilibrium_constants[i] : zero(T)
    end

    @inbounds for i = 1:reaction.n_reactions
        for j in reaction.i_reactant[i]
            kf[i] *= C[j]^reaction.reactant_orders[j, i]
        end
        if reaction.is_reversible[i]
            for j in reaction.i_product[i]
                kr[i] *= C[j]^reaction.product_stoich_coeffs[j, i]
            end
        end
        workspace.rates_of_progress[i] = kf[i] - kr[i]
    end

    if get_qdot
        return workspace.rates_of_progress
    end
    mul!(wdot, reaction.vk, workspace.rates_of_progress)
    return wdot
end
export wdot!

"compute reaction source term `dC/dt`"
function wdot_func(
    reaction,
    T,
    C,
    S0,
    h_mole;
    get_qdot=false,
    rate_multipliers=nothing,
    log_rate_data=nothing,
)
    multiplier_type = isnothing(rate_multipliers) ? Float64 : eltype(rate_multipliers)
    workspace_type = promote_type(
        typeof(T),
        eltype(C),
        eltype(S0),
        eltype(h_mole),
        multiplier_type,
    )
    workspace = KineticsWorkspace(reaction, workspace_type)
    wdot = zeros(workspace_type, size(reaction.vk, 1))
    return wdot!(
        wdot,
        reaction,
        T,
        C,
        S0,
        h_mole,
        workspace;
        get_qdot=get_qdot,
        rate_multipliers=rate_multipliers,
        log_rate_data=log_rate_data,
    )
end
export wdot_func

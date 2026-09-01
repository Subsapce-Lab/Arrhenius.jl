@inline function _plog_group_rate(plog::PlogData, group, T, logT)
    rate = zero(T)
    @inbounds for row in plog.rate_offsets[group]:(plog.rate_offsets[group + 1] - 1)
        A = plog.Arrhenius_coeffs[row, 1]
        b = plog.Arrhenius_coeffs[row, 2]
        Ea = plog.Arrhenius_coeffs[row, 3]
        rate += A * exp(b * logT - Ea * (4184.0 / R / T))
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
    tiny = 1.0e-300
    return exp(log(lower_rate + tiny) +
               fraction * (log(upper_rate + tiny) - log(lower_rate + tiny)))
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
)
    kf = workspace.kf
    kr = workspace.kr
    logT = log(T)
    activation_scale = 4184.0 / R / T
    @inbounds for i in eachindex(kf)
        kf[i] = reaction.Arrhenius_coeffs[i, 1] * exp(
            reaction.Arrhenius_coeffs[i, 2] * logT -
            reaction.Arrhenius_coeffs[i, 3] * activation_scale,
        )
    end

    if !isempty(reaction.plog.reaction_indices)
        P = sum(C) * R * T
        for (plog_index, reaction_index) in enumerate(reaction.plog.reaction_indices)
            @inbounds kf[reaction_index] =
                _plog_rate(reaction.plog, plog_index, T, P, logT)
        end
    end

    for i in reaction.index_three_body
        @inbounds kf[i] *= dot(@view(reaction.efficiencies_coeffs[:, i]), C)
    end

    for (j, i) in enumerate(reaction.index_falloff)
        @inbounds A0 = reaction.Arrhenius_0[j, 1]
        @inbounds b0 = reaction.Arrhenius_0[j, 2]
        @inbounds Ea0 = reaction.Arrhenius_0[j, 3]
        k0 = A0 * exp(b0 * logT - Ea0 * activation_scale)
        @inbounds collider = dot(@view(reaction.efficiencies_coeffs[:, i]), C)
        if collider <= zero(collider)
            kf[i] = zero(collider)
            continue
        end
        @inbounds Pr = k0 * collider / kf[i]
        if Pr <= zero(Pr)
            kf[i] = zero(Pr)
            continue
        end
        lPr = log10(Pr)
        @inbounds kf[i] *= Pr / (1 + Pr)

        if reaction.index_falloff_Troe[j] > 0
            k = reaction.index_falloff_Troe[j]
            @inbounds F_cent =
                (1 - reaction.Troe_[k, 1]) * exp(-T / reaction.Troe_[k, 4]) +
                reaction.Troe_[k, 1] * exp(-T / reaction.Troe_[k, 2]) +
                exp(-reaction.Troe_[k, 3] / T)
            lF_cent = log10(F_cent)
            C_troe = -0.4 - 0.67 * lF_cent
            N = 0.75 - 1.27 * lF_cent
            f1 = (lPr + C_troe) / (N - 0.14 * (lPr + C_troe))
            @inbounds kf[i] *= exp(log(10.0) * lF_cent / (1 + f1^2))
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
            workspace.delta_s[i] / R - workspace.delta_h[i] / (R * T) +
            log(one_atm / R / T) * reaction.vk_sum[i],
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
    )
end
export wdot_func

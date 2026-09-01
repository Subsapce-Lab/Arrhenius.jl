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

"compute reaction source term `dC/dt`"
function wdot_func(reaction, T, C, S0, h_mole; get_qdot=false)

    logT = log(T)
    @inbounds _kf = @. @view(reaction.Arrhenius_coeffs[:, 1]) * exp(
        @view(reaction.Arrhenius_coeffs[:, 2]) * logT -
        @view(reaction.Arrhenius_coeffs[:, 3]) * (4184.0 / R / T),
    )

    if !isempty(reaction.plog.reaction_indices)
        P = sum(C) * R * T
        for (plog_index, reaction_index) in enumerate(reaction.plog.reaction_indices)
            @inbounds _kf[reaction_index] =
                _plog_rate(reaction.plog, plog_index, T, P, logT)
        end
    end

    for i in reaction.index_three_body
        @inbounds _kf[i] *= dot(@view(reaction.efficiencies_coeffs[:, i]), C)
    end

    for (j, i) in enumerate(reaction.index_falloff)
        @inbounds A0, b0, Ea0 = reaction.Arrhenius_0[j, :]
        @inbounds k0 = A0 * exp(b0 * logT - Ea0 * 4184.0 / R / T)
        @inbounds collider = dot(@view(reaction.efficiencies_coeffs[:, i]), C)
        if collider <= zero(collider)
            _kf[i] = zero(collider)
            continue
        end
        @inbounds Pr = k0 * collider / _kf[i]
        if Pr <= zero(Pr)
            _kf[i] = zero(Pr)
            continue
        end
        lPr = log10(Pr)
        _kf[i] *= (Pr / (1 + Pr))

        # reference:
        # http://web.mit.edu/2.62/cantera/doc/html/classCantera_1_1Troe4.html#a38aa787421d426dfd0a587fd6fc8108e
        if reaction.index_falloff_Troe[j] > 0
            k = reaction.index_falloff_Troe[j]
            @inbounds F_cent =
                (1 - reaction.Troe_[k, 1]) * exp(-T / reaction.Troe_[k, 4]) +
                reaction.Troe_[k, 1] * exp(-T / reaction.Troe_[k, 2]) +
                exp(-reaction.Troe_[k, 3] / T)

            lF_cent = log10(F_cent)
            _C = -0.4 - 0.67 * lF_cent
            N = 0.75 - 1.27 * lF_cent
            @inbounds f1 = (lPr + _C) / (N - 0.14 * (lPr + _C))
            @inbounds _kf[i] *= exp(log(10.0) * lF_cent / (1 + f1^2))
        end
    end

    @inbounds ΔS_R = reaction.vk' * S0 / R
    @inbounds ΔH_RT = reaction.vk' * h_mole / (R * T)
    @inbounds Keq =
        @. exp(ΔS_R - ΔH_RT + log(one_atm / R / T) * reaction.vk_sum)
    @inbounds _kr = @. _kf / Keq * reaction.is_reversible

    for i = 1:reaction.n_reactions
        @inbounds for j in reaction.i_reactant[i]
            @inbounds _kf[i] *= C[j]^reaction.reactant_orders[j, i]
        end
        if reaction.is_reversible[i]
            @inbounds for j in reaction.i_product[i]
                @inbounds _kr[i] *= C[j]^reaction.product_stoich_coeffs[j, i]
            end
        end
    end

    if get_qdot
        return _kf - _kr
    end
    return reaction.vk * (_kf - _kr)
end
export wdot_func

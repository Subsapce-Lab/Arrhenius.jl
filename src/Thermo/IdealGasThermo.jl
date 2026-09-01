"""Struct for the ideal gas thermo.

nasa_low: Array with low temperature nasa coeff. for each species

nasa_high: Array with high temperature nasa coeff. for each species

Trange: Array with temperature ranges for each species

isTcommon: bool which indicates if both polynoms share same T at intersection

"""
struct IdealGasThermo <: Thermo 
    nasa_low::Array{Float64,2}
    nasa_high::Array{Float64,2}
    Trange::Array{Float64,2}
    isTcommon::Bool

end

@inline function _nasa7_coefficients(thermo::IdealGasThermo, species, T)
    return T <= thermo.Trange[species, 2] ? thermo.nasa_low : thermo.nasa_high
end

function cal_h_RT!(output, gas::Solution, thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    @inbounds for i in eachindex(output)
        nasa = _nasa7_coefficients(thermo, i, T)
        output[i] = nasa[i, 1] + nasa[i, 2] * T / 2 + nasa[i, 3] * T2 / 3 +
            nasa[i, 4] * T3 / 4 + nasa[i, 5] * T4 / 5 + nasa[i, 6] / T
    end
    return output
end
cal_h_RT!(output, gas::Solution, T, p, X) =
    cal_h_RT!(output, gas, gas.thermo, T, p, X)
export cal_h_RT!

function cal_s0_R!(output, gas::Solution, thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    logT = log(T)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    @inbounds for i in eachindex(output)
        nasa = _nasa7_coefficients(thermo, i, T)
        output[i] = nasa[i, 1] * logT + nasa[i, 2] * T + nasa[i, 3] * T2 / 2 +
            nasa[i, 4] * T3 / 3 + nasa[i, 5] * T4 / 4 + nasa[i, 7]
    end
    return output
end
cal_s0_R!(output, gas::Solution, T, p, X) =
    cal_s0_R!(output, gas, gas.thermo, T, p, X)
export cal_s0_R!

function cal_cp_R!(output, gas::Solution, thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    @inbounds for i in eachindex(output)
        nasa = _nasa7_coefficients(thermo, i, T)
        output[i] = nasa[i, 1] + nasa[i, 2] * T + nasa[i, 3] * T2 +
            nasa[i, 4] * T3 + nasa[i, 5] * T4
    end
    return output
end
cal_cp_R!(output, gas::Solution, T, p, X) =
    cal_cp_R!(output, gas, gas.thermo, T, p, X)
export cal_cp_R!
"""
Constructor for the idealGasThermo:

yaml:: Dict of the input yaml file
"""
function IdealGasThermo(yaml::AbstractDict)
    n_species, n_reactions, species_names,
    elements, n_elements, ele_matrix = read_species_basics(yaml)

    nasa_low = zeros(n_species, 7)
    nasa_high = zeros(n_species, 7)
    Trange = zeros(n_species, 3)

    _species_names =
        [yaml["species"][i]["name"] for i = 1:length(yaml["species"])]

    for (i, species) in enumerate(species_names)
        spec = yaml["species"][findfirst(x -> x == species, _species_names)]
        thermo = spec["thermo"]
        data = thermo["data"]
        ranges = Float64.(thermo["temperature-ranges"])
        if length(data) == 1 && length(ranges) == 2
            # A single NASA7 region is valid Cantera YAML. Duplicating the
            # polynomial preserves its value on both sides of an arbitrary
            # internal split while retaining the existing two-region layout.
            nasa_low[i, :] = data[1]
            nasa_high[i, :] = data[1]
            midpoint = clamp(1000.0, ranges[1], ranges[2])
            Trange[i, :] .= (ranges[1], midpoint, ranges[2])
        elseif length(data) == 2 && length(ranges) == 3
            nasa_low[i, :] = data[1]
            nasa_high[i, :] = data[2]
            Trange[i, :] .= ranges
        else
            throw(ArgumentError(
                "species $species must define one or two NASA7 regions; " *
                "found $(length(data)) data regions and $(length(ranges)) bounds",
            ))
        end

        for j = 1:n_elements
            if haskey(spec["composition"], elements[j])
                ele_matrix[j, i] = spec["composition"][elements[j]]
            end
        end
    end
    isTcommon = (maximum(Trange[:, 2]) - minimum(Trange[:, 2])) < 0.01
    return  IdealGasThermo(nasa_low, nasa_high, Trange, isTcommon)
end
"""
    cal_h_RT(gas, T, p, X)

calculates the dimensionless mole based enthalpy (h) for each species
"""
function cal_h_RT(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    H_T = [1.0, T / 2.0, T^2 / 3.0, T^3 / 4.0, T^4 / 5.0, 1.0 / T]
    if thermo.isTcommon && T <= thermo.Trange[1, 2]
        h_mole = @view(thermo.nasa_low[:, 1:6]) * H_T 
    elseif thermo.isTcommon
        h_mole = @view(thermo.nasa_high[:, 1:6]) * H_T 
    else
        h_mole = @view(thermo.nasa_high[:, 1:6]) * H_T
        use_low = T .<= @view(thermo.Trange[:, 2])
        h_mole[use_low] .= @view(thermo.nasa_low[use_low, 1:6]) * H_T
    end
    # H_mole = dot(h_mole, X)
    return h_mole
end

"""
    cal_s0_R(gas, T, p, X)

calculates the dimensionless mole based reference state entropy (s0) for each species
"""
function cal_s0_R(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    S_T = [log(T), T, T^2 / 2.0, T^3 / 3.0, T^4 / 4.0, 1.0]
    if thermo.isTcommon && T <= thermo.Trange[1, 2]
        S0 = @view(thermo.nasa_low[:, [1, 2, 3, 4, 5, 7]]) * S_T 
    elseif thermo.isTcommon
        S0 = @view(thermo.nasa_high[:, [1, 2, 3, 4, 5, 7]]) * S_T 
    else
        S0 = @view(thermo.nasa_high[:, [1, 2, 3, 4, 5, 7]]) * S_T
        use_low = T .<= @view(thermo.Trange[:, 2])
        S0[use_low] .=
            @view(thermo.nasa_low[use_low, [1, 2, 3, 4, 5, 7]]) * S_T
    end
    return S0 
end
export cal_s0_R

"""
    cal_s_R(gas, T, p, X)

calculates the dimensionless mole based entropy (s) for each species
"""
function cal_s_R(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
   return cal_s0_R(gas,thermo, T,p,X) - log.(max.(X,1e-30)) .- log(p/one_atm)
end

"""
    cal_g_RT(gas, T, p, X)

calculates the dimensionless mole based free gibbs energy (g) for each species
"""
function cal_g_RT(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    return cal_h_RT(gas,thermo, T, p, X) - cal_s_R(gas,thermo, T, p, X)
end

"""
    cal_u_RT(gas, T, p, X)

calculates the dimensionless mole based internal energy (u) for each species
"""
function cal_u_RT(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    return cal_h_RT(gas,thermo, T, p, X) .- 1
end

"""
    cal_a_RT(gas, T, p, X)

calculates the dimensionless mole based helmholz free energy (a) for each species
"""
function cal_a_RT(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    return cal_u_RT(gas,thermo, T, p, X) - cal_s_R(gas,thermo, T, p, X)
end

"""
    cal_cp_R(gas, T, p, X)

calculates the dimensionless mole based heat capacity 
at constant pressure (cp) for each species
"""
function cal_cp_R(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    cp_T = [1.0, T, T^2, T^3, T^4]
    if thermo.isTcommon && T <= thermo.Trange[1, 2]
        cp = @view(thermo.nasa_low[:, 1:5]) * cp_T
    elseif thermo.isTcommon
        cp = @view(thermo.nasa_high[:, 1:5]) * cp_T
    else
        cp = @view(thermo.nasa_high[:, 1:5]) * cp_T
        use_low = T .<= @view(thermo.Trange[:, 2])
        cp[use_low] .= @view(thermo.nasa_low[use_low, 1:5]) * cp_T
    end
    return cp
end

"""
    cal_cv_R(gas, T, p, X)

calculates the dimensionless mole based heat capacity 
at constant volume (cv) for each species
"""
function cal_cv_R(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
   return cal_cp_R(gas,thermo, T, p, X) .- 1
end


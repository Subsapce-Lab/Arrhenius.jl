# include all different thermo files here 
#TODO Automate it to include all jl files in the Thermo folder
include("Thermo/IdealGasThermo.jl")
# Metaprogramming loop to generate and export all mass and mean functions
property_names=((:cv,"Heat capacity at constant volume (cv)"),
                (:cp,"Heat capacity at constant pressure (cp)"), 
                (:s,"entropy (s)"), 
                (:s0,"reference entropy (s0)"), 
                (:h,"enthalpy (h)"),
                (:a,"helmholz free energy (a)"), 
                (:g,"gibbs free energy (g)"), 
                (:u,"internal energy"))
for (phi, doc_name) in property_names
    # decide if dimless factor is R or RT
    if phi in (:cv, :cp, :s, :s0)
        dimmless = Symbol(:_R)
        RRT =:(R)
    else
        dimmless = Symbol(:_RT)
        RRT =:(R*T)
    end
    #define the 5 functions for each quantity
    cal_phi_dimless = Symbol(:cal_,phi,dimmless)
    cal_phi = Symbol(:cal_,phi)
    cal_phimass =  Symbol(:cal_,phi,:mass)
    cal_phi_mean = Symbol(cal_phi,:_mean)
    cal_phimass_mean = Symbol(cal_phimass,:_mean)

    @eval begin 
        # Dispatches the call from a solution object to it's thermo object
        $cal_phi_dimless(gas::Arrhenius.Solution, T, p, X)=$cal_phi_dimless(gas,gas.thermo, T, p, X)
        export $cal_phi_dimless 
        """
            $($cal_phi)(Solution, T, p, X)

        calculates the molar $($doc_name) for each species
        """
        function $cal_phi(gas::Arrhenius.Solution,thermo::Thermo,
                           T::Real, p::Real, X::AbstractArray)
            return $cal_phi_dimless(gas,thermo, T, p, X) * $RRT
        end
        $cal_phi(gas::Arrhenius.Solution, T, p, X)=$cal_phi(gas,gas.thermo, T, p, X)
        export $cal_phi 
        """
            $($cal_phi_mean)(Solution, T, p, X)
        
        calculates the mean mole based $($doc_name) of the mixture
        """
        function $cal_phi_mean(gas::Arrhenius.Solution,thermo::Thermo,
                                T::Real, p::Real, X::AbstractArray)
            return dot(X,$cal_phi_dimless(gas,thermo, T, p, X)) * $RRT
        end
        $cal_phi_mean(gas::Arrhenius.Solution, T, p, X)=$cal_phi_mean(gas,gas.thermo, T, p, X)
        export $cal_phi_mean 
        """
            $($cal_phimass)(Solution, T, p, X)

        calculates the partial mass based $($doc_name) for each species
        """
        function $cal_phimass(gas::Arrhenius.Solution,thermo::Thermo,
                              T::Real, p::Real, X::AbstractArray)
            return $cal_phi(gas,thermo, T, p, X) ./gas.MW
        end
        $cal_phimass(gas::Arrhenius.Solution, T, p, X)=$cal_phimass(gas,gas.thermo, T, p, X)
        export $cal_phimass 
        """
            $($cal_phimass_mean)(Solution, T, p, X)

        calculates the mean mass based $($doc_name) of the mixture
        """
        function $cal_phimass_mean(gas::Arrhenius.Solution,thermo::Thermo,
                                    T::Real, p::Real, X::AbstractArray)
            return $cal_phi_mean(gas,thermo, T, p, X) / dot(X,gas.MW)
        end        
        $cal_phimass_mean(gas::Arrhenius.Solution, T, p, X)=$cal_phimass_mean(gas,gas.thermo, T, p, X)
        export $cal_phimass_mean 
    end
end


# ------ Deprecated ----------# 

"get specific of heat capacity"
function get_cp(gas, T, X, mean_MW)
    cp = cal_cp_R(gas, T, one_atm, X)
    cp_mole = dot(cp, X) * R
    cp_mass = cp_mole / mean_MW
    return cp_mole, cp_mass
end
export get_cp


"get specific of heat capacity"
function get_cv(cp_mole, cp_mass, mean_MW)
    cv_mole = cp_mole - R
    cv_mass = cv_mole / mean_MW
    return cv_mole, cv_mass
end
export get_cv


"get enthaphy (H) per mole"
function get_H(gas, T, Y, X)
    return cal_h_RT(gas, T, one_atm, X) * R * T
end
export get_H


"get enthaphy (H) per mass"
function H_mass_func(gas, h_mole, Y)
    return dot(h_mole ./ gas.MW, Y)
end
export H_mass_func


"get enthaphy (U) per mole"
function get_U(h_mole, T)
    u_mole = h_mole .- (R * T)
    return u_mole
end
export get_U


"get enthaphy (U) per mass"
function U_mass_func(gas, u_mole, Y)
    return dot(u_mole ./ gas.MW, Y)
end
export U_mass_func


"get entropy (S)"
function get_S(gas, T, P, X)
    return cal_s0_R(gas, T, P, X) * R
end
export get_S

"get entropy (S) per unit mass"
function S_mass_func(gas, s_mole, Y)
    return dot(s_mole ./ gas.MW, Y)
end
export S_mass_func

"""
    PlasmaEnergyReactor(state; volume=1.0)

A closed, constant-pressure Boltzmann-plasma reactor. Its state is
`[mass, total_enthalpy, mass_fractions...]`; electron temperature and the
electric field/EEDF are held fixed between explicit [`update_eedf!`](@ref)
calls on the prepared RHS.
"""
struct PlasmaEnergyReactor
    mechanism::PlasmaMechanism
    pressure::Float64
    electron_temperature::Float64
    mean_electron_energy::Float64
    electric_field::Float64
    eedf::EEDFResult
    temperature::Float64
    mass::Float64
    total_enthalpy::Float64
    mass_fractions::Vector{Float64}
end

function PlasmaEnergyReactor(s::PlasmaState;volume=1.0)
    s.mechanism.thermal===nothing &&
        throw(ArgumentError("PlasmaEnergyReactor requires a Boltzmann plasma mechanism"))
    s.eedf===nothing && throw(ArgumentError("call update_eedf! before constructing the reactor"))
    V=_plasma_positive(volume,"volume")
    p=plasma_properties(s)
    thermo=plasma_thermodynamics(s)
    mass=s.density*V
    return PlasmaEnergyReactor(s.mechanism,p.P,p.Te,s.mean_electron_energy,
        s.electric_field,deepcopy(s.eedf),s.temperature,mass,mass*thermo.h_mass,
        copy(s.mass_fractions))
end

reactor_state(r::PlasmaEnergyReactor)=vcat(r.mass,r.total_enthalpy,r.mass_fractions)

mutable struct PlasmaEnergyRHS{W}
    reactor::PlasmaEnergyReactor
    workspace::W
    eedf::EEDFResult
    electric_field::Float64
    mobility::Float64
    last_temperature::Float64
    jac_state::Vector{Float64}
    jac_base::Vector{Float64}
    jac_plus::Vector{Float64}
    jac_minus::Vector{Float64}
end

function reactor_rhs(r::PlasmaEnergyReactor)
    u=reactor_state(r)
    w=_PlasmaThermoRateWorkspace(r.mechanism,Float64)
    _plasma_collision_rates!(w.collision_rates,w.collision_integrand,r.mechanism,r.eedf)
    return PlasmaEnergyRHS(r,w,deepcopy(r.eedf),r.electric_field,r.eedf.mobility,
        r.temperature,copy(u),similar(u),similar(u),similar(u))
end

function _plasma_hp_temperature!(w,m,target,Y,Te,guess;rtol=1e-13,maxiter=500)
    isfinite(target) || throw(DomainError(target,"finite specific enthalpy required"))
    T=guess
    e=m.electron_index
    for _ in 1:maxiter
        isfinite(T) && T>0 || throw(DomainError(T,"positive finite gas temperature required"))
        _plasma_species_thermo!(w.h,w.cp,w.s0,m,T,Te)
        value=zero(T)
        scale=one(T)
        slope=zero(T)
        @inbounds for k in eachindex(Y)
            amount=Y[k]/m.MW[k]
            term=amount*w.h[k]
            value+=term
            scale+=abs(term)
            k==e || (slope+=amount*w.cp[k])
        end
        residual=target-value
        abs(residual)<=rtol*max(abs(target),scale) && return T
        isfinite(slope) && slope>0 ||
            throw(DomainError(slope,"positive heavy-species heat capacity required during plasma HP inversion"))
        step=clamp(residual/slope,-100.0,100.0)
        T+=max(step,-0.5T)
    end
    throw(ErrorException("plasma HP inversion did not converge in $maxiter iterations"))
end

function _plasma_energy_state!(rhs::PlasmaEnergyRHS,u)
    r,m,w=rhs.reactor,rhs.reactor.mechanism,rhs.workspace
    length(u)==m.n_species+2 || throw(DimensionMismatch("energy-plasma state must contain mass, total enthalpy, and all species"))
    mass=u[1]
    isfinite(mass) && mass>0 || throw(DomainError(mass,"positive finite reactor mass required"))
    H=u[2]
    isfinite(H) || throw(DomainError(H,"finite total enthalpy required"))
    Y=@view u[3:end]
    all(isfinite,Y) || throw(DomainError(Y,"finite signed mass fractions required"))
    T=_plasma_hp_temperature!(w,m,H/mass,Y,r.electron_temperature,rhs.last_temperature)
    rhs.last_temperature=T
    inverse_mw=zero(T)
    denominator=zero(T)
    @inbounds for k in eachindex(Y)
        amount=Y[k]/m.MW[k]
        inverse_mw+=amount
        denominator+=(k==m.electron_index ? r.electron_temperature : T)*amount
    end
    isfinite(inverse_mw) && inverse_mw>0 || throw(DomainError(inverse_mw,"positive finite inverse molecular weight required"))
    isfinite(denominator) && denominator>0 || throw(DomainError(denominator,"positive finite two-temperature EOS denominator required"))
    rho=r.pressure/(R*denominator)
    @inbounds for k in eachindex(Y)
        w.X[k]=(Y[k]/m.MW[k])/inverse_mw
    end
    return T,rho,inverse_mw,Y
end

function (rhs::PlasmaEnergyRHS)(du,u,p,t)
    m=rhs.reactor.mechanism
    length(du)==m.n_species+2 || throw(DimensionMismatch("derivative must match energy-plasma state"))
    T,rho,_,Y=_plasma_energy_state!(rhs,u)
    _plasma_thermal_sources!(rhs.workspace,m,T,rhs.reactor.electron_temperature,
        rhs.reactor.pressure,rho,Y)
    du[1]=0.0
    ye=Y[m.electron_index]
    if ye>0 && rhs.mobility>0 && rhs.electric_field>0
        factor=_EEDF_ELECTRON_CHARGE*_EEDF_AVOGADRO_KMOL/m.MW[m.electron_index]
        du[2]=u[1]*ye*factor*rhs.mobility*rhs.electric_field^2
    else
        du[2]=0.0
    end
    @inbounds for k in eachindex(Y)
        du[k+2]=m.MW[k]*rhs.workspace.wdot[k]/rho
    end
    return nothing
end
(rhs::PlasmaEnergyRHS)(du,u)=rhs(du,u,nothing,0.0)

function reactor_properties(rhs::PlasmaEnergyRHS,u)
    r,m,w=rhs.reactor,rhs.reactor.mechanism,rhs.workspace
    T,rho,inverse_mw,Y=_plasma_energy_state!(rhs,u)
    h_mass=zero(T)
    @inbounds for k in eachindex(Y)
        h_mass+=(Y[k]/m.MW[k])*w.h[k]
    end
    return (T=T,Te=r.electron_temperature,P=r.pressure,rho=rho,X=copy(w.X),Y=copy(Y),
        mass=u[1],h_mass=h_mass,volume=u[1]/rho,electric_field=rhs.electric_field,
        reduced_electric_field=rhs.electric_field/(rho*inverse_mw*_EEDF_AVOGADRO_KMOL),
        electron_mobility=rhs.mobility,mean_electron_energy=r.mean_electron_energy,
        mass_fraction_sum=sum(Y),elemental_inventory=m.elemental_matrix*(Y./m.MW))
end
reactor_properties(r::PlasmaEnergyReactor,u=reactor_state(r))=reactor_properties(reactor_rhs(r),u)

"""Refresh the accepted-state EEDF and cached collision rate coefficients."""
function update_eedf!(rhs::PlasmaEnergyRHS,u;reduced_field=nothing,options=TwoTermOptions())
    p=reactor_properties(rhs,u)
    m=rhs.reactor.mechanism
    N=p.rho*sum(p.Y./m.MW)*_EEDF_AVOGADRO_KMOL
    isfinite(N) && N>0 || throw(DomainError(N,"positive finite number density required"))
    changing_field=reduced_field!==nothing
    EN=changing_field ? Float64(reduced_field) : rhs.electric_field/N
    isfinite(EN) && EN>=0 || throw(ArgumentError("E/N must be finite and nonnegative"))
    state=_accepted_eedf_state(m.thermal.eedf_model;T=p.T,P=p.P,
        mole_fractions=Dict(zip(m.species_names,p.X)),
        molecular_weights=Dict(zip(m.species_names,m.MW)),reduced_field=EN,number_density=N)
    result=_solve_accepted_eedf(m.thermal.eedf_model,state;options=options,initial=rhs.eedf)
    result.converged || throw(ErrorException("Boltzmann EEDF did not converge"))
    rhs.eedf=result
    rhs.mobility=result.mobility
    changing_field && (rhs.electric_field=EN*N)
    _plasma_collision_rates!(rhs.workspace.collision_rates,
        rhs.workspace.collision_integrand,m,result)
    return rhs
end

function reactor_jacobian!(J,u,rhs::PlasmaEnergyRHS,t=0)
    n=length(u)
    size(J)==(n,n) || throw(DimensionMismatch("Jacobian must have state dimensions"))
    copyto!(rhs.jac_state,u)
    rhs(rhs.jac_base,u,nothing,t)
    rel=cbrt(eps(Float64))
    e=rhs.reactor.mechanism.electron_index+2
    @inbounds for j in 1:n
        scale=j==1 ? 1.0 : j==2 ? 1.0 : j==e ? 1e-16 : 1e-12
        step=rel*max(abs(u[j]),scale)
        if j==1 && u[j]<=step
            rhs.jac_state[j]=u[j]+step
            rhs(rhs.jac_plus,rhs.jac_state,nothing,t)
            rhs.jac_state[j]=u[j]+2step
            rhs(rhs.jac_minus,rhs.jac_state,nothing,t)
            for i in 1:n
                J[i,j]=(-3rhs.jac_base[i]+4rhs.jac_plus[i]-rhs.jac_minus[i])/(2step)
            end
        else
            rhs.jac_state[j]=u[j]+step
            rhs(rhs.jac_plus,rhs.jac_state,nothing,t)
            rhs.jac_state[j]=u[j]-step
            rhs(rhs.jac_minus,rhs.jac_state,nothing,t)
            for i in 1:n
                J[i,j]=(rhs.jac_plus[i]-rhs.jac_minus[i])/(2step)
            end
        end
        rhs.jac_state[j]=u[j]
    end
    fill!(view(J,1,:),0.0)
    fill!(view(J,2,:),0.0)
    Y=@view u[3:end]
    if Y[rhs.reactor.mechanism.electron_index]>0 && rhs.mobility>0 && rhs.electric_field>0
        factor=_EEDF_ELECTRON_CHARGE*_EEDF_AVOGADRO_KMOL/
            rhs.reactor.mechanism.MW[rhs.reactor.mechanism.electron_index]*rhs.mobility*rhs.electric_field^2
        J[2,1]=Y[rhs.reactor.mechanism.electron_index]*factor
        J[2,e]=u[1]*factor
    end
    rhs(rhs.jac_base,u,nothing,t)
    copyto!(rhs.jac_state,u)
    return nothing
end

function reactor_problem(r::PlasmaEnergyReactor,tspan)
    length(tspan)==2 && all(isfinite,tspan) && tspan[2]>tspan[1] ||
        throw(ArgumentError("tspan must be finite and strictly increasing"))
    rhs=reactor_rhs(r)
    jac=(J,u,p,t)->reactor_jacobian!(J,u,rhs,t)
    tgrad=(du,u,p,t)->(fill!(du,zero(eltype(du)));nothing)
    return (f=rhs,jac=jac,tgrad=tgrad,u0=reactor_state(r),
        tspan=(float(tspan[1]),float(tspan[2])),p=nothing)
end

solve_reactor(r::PlasmaEnergyReactor,tspan;integrator,kwargs...)=
    integrator(reactor_problem(r,tspan);kwargs...)

export PlasmaEnergyReactor,PlasmaEnergyRHS

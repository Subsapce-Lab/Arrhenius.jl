"""Planar premixed flame with signed charged species and an algebraic electric field.

Construct through FreeFlame or BurnerFlame with an ionized-gas mechanism.
The initial stage freezes charged diffusion. Enable it with
set_electric_field!(flame, true) before the second solve!.
"""
mutable struct IonizedFlame{G<:Solution} <: AbstractPremixedFlame
    gas::G
    grid::Vector{Float64}
    pressure::Float64
    inlet_temperature::Float64
    inlet_Y::Vector{Float64}
    state::Matrix{Float64}
    anchor::Int
    fixed_temperature::Float64
    converged::Bool
    kind::Symbol
    mass_flux::Float64
    field_enabled::Bool
    ion_data::IonTransportData
end

const _ION_FLAME_FARADAY = _ION_QE*6.02214076e26
const _ION_FLAME_EPS0 = inv(299792458.0^2 *
    (2*7.2973525693e-3*6.62607015e-34/(_ION_QE^2*299792458.0)))

function IonizedFlame(gas::Solution;kind=:free,T=300.0,P=one_atm,X,
        width=.05,grid=nothing,mdot=nothing)
    kind in (:free,:burner) || throw(ArgumentError("kind must be :free or :burner"))
    isfinite(T) && 200<=T<=6000 && isfinite(P) && P>0 ||
        throw(ArgumentError("finite temperature in 200-6000 K and positive pressure required"))
    isfinite(width) && width>0 || throw(ArgumentError("positive finite width required"))
    kind==:burner && !(mdot isa Real && isfinite(mdot) && mdot>0) &&
        throw(ArgumentError("burner mass flux must be finite and positive"))
    data=IonTransportData(gas)
    x=mole_fractions(gas,X);y=x.*gas.MW./dot(x,gas.MW)
    eq=equilibrate(gas;T,P,X=x,mode=:HP)
    rho=P*dot(x,gas.MW)/(R*T)
    massflux=kind==:burner ? Float64(mdot) : rho
    z=isnothing(grid) ? width.*[0,.2,.3,.35,.4,.5,.6,.8,1] : Float64.(grid)
    transition=.2*width
    if isnothing(grid) && kind==:burner
        _,lambda,_=ionized_transport!(IonTransportWorkspace(data),data,P,eq.T,eq.X)
        cp=dot(eq.Y,cal_cp_R(gas,eq.T,P,eq.X).*R./gas.MW)
        ell=min(4lambda/(massflux*cp),width/16)
        z=unique(vcat(ell.*[0,.5,1,2,4,8,16],Float64(width)))
        transition=2ell
    end
    length(z)>=5 && all(isfinite,z) && all(>(0),diff(z)) ||
        throw(ArgumentError("at least five increasing finite grid coordinates required"))
    span=z[end]-z[1];n=gas.n_species;u=zeros(n+3,length(z))
    for j in eachindex(z)
        fraction=kind==:free ? clamp(((z[j]-z[1])/span-.3)/.2,0,1) :
            clamp((z[j]-z[1])/transition,0,1)
        temp=T+fraction*(eq.T-T)
        u[1,j]=temp/1000
        for k in 1:n
            u[k+1,j]=y[k]+fraction*(eq.Y[k]-y[k])
        end
        inverseMW=sum(u[k+1,j]/gas.MW[k] for k in 1:n)
        u[end,j]=massflux*R*temp*inverseMW/P
    end
    anchor=clamp(argmin(abs.(u[1,:].-(.75*T+.25*eq.T)/1000)),2,length(z)-1)
    IonizedFlame(gas,z,Float64(P),Float64(T),y,u,anchor,1000*u[1,anchor],
        false,kind,massflux,false,data)
end

"""Enable or freeze charged diffusion and the electric-field equation for the next solve."""
function set_electric_field!(f::IonizedFlame,enabled::Bool)
    f.field_enabled=enabled
    f.converged=false
    return f
end
electric_field(f::IonizedFlame)=1000 .* vec(f.state[end-1,:])
mass_fractions(f::IonizedFlame)=copy(f.state[2:f.gas.n_species+1,:])
velocity(f::IonizedFlame)=copy(vec(f.state[end,:]))

include("IonFlameProperties.jl")
include("IonFlameResidual.jl")

export IonizedFlame, set_electric_field!, electric_field

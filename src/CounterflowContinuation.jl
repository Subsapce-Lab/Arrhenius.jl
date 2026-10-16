"""Fixed-grid pseudo-arclength predictor-corrector for counterflow flames.
Supported parameters: fuel_mass_flux, oxidizer_mass_flux, pressure, or inlet_flux_scale (proportional scaling of both inlet mass fluxes).
The bordered correction retains the banded flame Jacobian and uses a scalar
Schur complement. The initial flame must be converged; no grid refinement is
performed during tracing.
"""
struct CounterflowContinuationPoint
    parameter::Float64
    max_temperature::Float64
    max_strain_rate::Float64
    state::Matrix{Float64}
    residual_inf::Float64
    arclength::Float64
    turning_point::Bool
end

function _continuation_strain_rate(f,u)
    n=f.gas.n_species
    v=[u[end,j]*R*(1000u[1,j])*sum(u[k+1,j]/f.gas.MW[k] for k in 1:n)/f.pressure
        for j in axes(u,2)]
    return maximum(abs.(diff(v)./diff(f.grid)))
end

function _continuation_parameter(f,name::Symbol,fluxbase=nothing)
    name in (:fuel_mass_flux,:oxidizer_mass_flux,:pressure,:inlet_flux_scale) ||
        throw(ArgumentError("parameter must be :fuel_mass_flux, :oxidizer_mass_flux, :pressure, or :inlet_flux_scale"))
    isnothing(f.control_points) || throw(ArgumentError("disable two-point control before pseudo-arclength continuation"))
    if name===:inlet_flux_scale
        isnothing(fluxbase) && throw(ArgumentError("internal inlet flux baseline is missing"))
        scale=f.fuel_mass_flux/fluxbase[1]
        isapprox(f.oxidizer_mass_flux/fluxbase[2],scale;rtol=1e-10,atol=0) ||
            throw(ArgumentError("inlet mass fluxes do not share a common scale"))
        return Float64(scale)
    end
    Float64(getproperty(f,name))
end
function _continuation_set_parameter!(f,name::Symbol,value,fluxbase=nothing)
    isfinite(value) && value>0 || throw(ArgumentError("continuation parameter must remain positive and finite"))
    if name===:inlet_flux_scale
        isnothing(fluxbase) && throw(ArgumentError("internal inlet flux baseline is missing"))
        f.fuel_mass_flux=fluxbase[1]*value; f.oxidizer_mass_flux=fluxbase[2]*value
    else
        setproperty!(f,name,Float64(value))
    end
    f.converged=false; value
end

function _continuation_border_solve!(f,w,u,p,parameter,r,fp,rhs,z,analytic=true,fluxbase=nothing)
    _continuation_set_parameter!(f,parameter,p,fluxbase)
    counterflow_residual!(r,f,u,w)
    _counterflow_jacobian!(f,u,w,r;analytic)
    bw=2size(u,1)-1
    _,piv=LinearAlgebra.LAPACK.gbtrf!(bw,bw,length(u),w.band)
    copyto!(rhs,vec(r)); rhs .*= -1
    LinearAlgebra.LAPACK.gbtrs!('N',bw,bw,length(u),w.band,piv,rhs)
    dp=sqrt(eps(Float64))*max(abs(p),1.0)
    _continuation_set_parameter!(f,parameter,p+dp,fluxbase)
    counterflow_residual!(fp,f,u,w); fp .= (fp .- r)./dp
    _continuation_set_parameter!(f,parameter,p,fluxbase)
    copyto!(z,vec(fp)); LinearAlgebra.LAPACK.gbtrs!('N',bw,bw,length(u),w.band,piv,z)
    return r
end

function _continuation_correct!(f,w,u0,p0,tu,tp,parameter,weights,qscale,ds;
        tolerance=1e-8,maxiters=18,analytic=true,fluxbase=nothing)
    u=copy(u0); p=p0; B,N=size(u); r=similar(u); fp=similar(u)
    rhs=Vector{Float64}(undef,length(u)); z=similar(rhs)
    pred=u0 .+ ds.*tu; ppred=p0+ds*tp
    cu=tu ./ reshape(weights.^2,B,1); cp=tp/qscale^2
    merit0=Inf
    for it in 1:maxiters
        _continuation_border_solve!(f,w,u,p,parameter,r,fp,rhs,z,analytic,fluxbase)
        g=sum(tu .* (u .- pred) ./ reshape(weights.^2,B,1))+tp*(p-ppred)/qscale^2
        rn=norm(r,Inf)
        rn<=tolerance && abs(g)<=tolerance*max(1.,ds) && return (u,p,rn,it)
        denom=cp-dot(vec(cu),z)
        abs(denom)>eps(Float64)*max(1.,abs(cp)) || return nothing
        dp=(-g-dot(vec(cu),rhs))/denom
        du=reshape(rhs .- z.*dp,B,N); accepted=false; alpha=1.
        for _ in 1:8
            un=u .+ alpha.*du; pn=p+alpha*dp
            if pn>0
                _continuation_set_parameter!(f,parameter,pn,fluxbase); counterflow_residual!(fp,f,un,w)
                gn=sum(tu .* (un .- pred) ./ reshape(weights.^2,B,1))+tp*(pn-ppred)/qscale^2
                merit=hypot(norm(fp,Inf),abs(gn))
                if isfinite(merit) && (merit<merit0 || merit<=(1-1e-4*alpha)*hypot(rn,abs(g)))
                    u=un; p=pn; merit0=merit; accepted=true; break
                end
            end
            alpha*=.5
        end
        accepted || return nothing
    end
    return nothing
end

function _continuation_tangent!(f,w,u,p,parameter,weights,qscale;analytic=true,fluxbase=nothing)
    r=similar(u); fp=similar(u); rhs=Vector{Float64}(undef,length(u)); z=similar(rhs)
    _continuation_border_solve!(f,w,u,p,parameter,r,fp,rhs,z,analytic,fluxbase)
    du=reshape(-z,size(u)); tp=1.
    nt=sqrt(sum((du./weights).^2)+(tp/qscale)^2); du./=nt; tp/=nt
    return du,tp
end

"""Trace a fixed-grid branch with a bordered pseudo-arclength Newton corrector.
`parameter` is `:fuel_mass_flux`, `:oxidizer_mass_flux`, `:pressure`, or `:inlet_flux_scale`.
Returns points with states, residual infinity norms, arclengths, and turning flags. `parameter_scale` and `state_scale`
control the relative continuation metric; both must be positive and finite. The initial state is used to construct
componentwise state scales, multiplied by `state_scale`.
"""
function pseudo_arclength!(f;parameter=:fuel_mass_flux,steps=40,ds=.05,
        min_ds=ds/16,max_ds=2ds,tolerance=1e-8,max_corrector_iters=18,
        parameter_scale=nothing,state_scale=1.,analytic=true)
    parameter=Symbol(parameter)
    (f isa CounterflowDiffusionFlame || f isa CounterflowPremixedFlame || f isa CounterflowTwinPremixedFlame) ||
        throw(ArgumentError("pseudo-arclength is implemented for diffusion, opposed premixed, and twin premixed flames"))
    fluxbase=parameter===:inlet_flux_scale ? (f.fuel_mass_flux,f.oxidizer_mass_flux) : nothing
    steps>=1 && isfinite(ds) && ds>0 && 0<min_ds<=ds<=max_ds ||
        throw(ArgumentError("require steps>=1 and 0<min_ds<=ds<=max_ds"))
    tolerance>0 && isfinite(tolerance) || throw(ArgumentError("positive finite tolerance required"))
    f.converged || throw(ArgumentError("initial flame must be converged"))
    p0=_continuation_parameter(f,parameter,fluxbase); u0=copy(f.state)
    qscale=isnothing(parameter_scale) ? max(abs(p0),1.) : Float64(parameter_scale)
    isfinite(qscale) && qscale>0 || throw(ArgumentError("parameter_scale must be positive and finite"))
    isfinite(state_scale) && state_scale>0 || throw(ArgumentError("state_scale must be positive and finite"))
    weights=state_scale .* [max(.05*maximum(abs,@view(u0[k,:])),1e-7) for k in axes(u0,1)]; w=CounterflowWorkspace(f)
    tu,tp=_continuation_tangent!(f,w,u0,p0,parameter,weights,qscale;analytic,fluxbase)
    pts=CounterflowContinuationPoint[]; arc=0.; lastsign=sign(tp); step=Float64(ds)
    r=similar(u0); counterflow_residual!(r,f,u0,w)
    push!(pts,CounterflowContinuationPoint(p0,maximum(temperature(f)),_continuation_strain_rate(f,u0),copy(u0),norm(r,Inf),arc,false))
    for _ in 1:steps
        corrected=nothing
        while corrected===nothing
            corrected=_continuation_correct!(f,w,u0,p0,tu,tp,parameter,weights,qscale,step;
                tolerance,maxiters=max_corrector_iters,analytic,fluxbase)
            if corrected===nothing
                step*=.5
                if step<min_ds
                    _continuation_set_parameter!(f,parameter,p0,fluxbase)
                    f.state .= u0; f.converged=true
                    error("pseudo-arclength corrector failed at min_ds=$min_ds")
                end
            end
        end
        u1,p1,rn,nit=corrected; arc+=step
        turn=sign(p1-p0)!=0 && sign(p1-p0)!=lastsign
        du=u1.-u0; dp=p1-p0
        orient=sum(du.*tu./reshape(weights.^2,size(u0,1),1))+dp*tp/qscale^2
        orient<0 && (du .*= -1; dp *= -1)
        nt=sqrt(sum((du./weights).^2)+(dp/qscale)^2); tu=du./nt; tp=dp/nt
        _continuation_set_parameter!(f,parameter,p1,fluxbase); f.state .= u1; f.converged=true
        push!(pts,CounterflowContinuationPoint(p1,maximum(temperature(f)),_continuation_strain_rate(f,u1),copy(u1),rn,arc,turn))
        u0=copy(u1); p0=p1; lastsign=sign(tp)
        step=min(max_ds,step*(nit<=5 ? 1.25 : nit>=12 ? .7 : 1.0))
    end
    return pts
end
export CounterflowContinuationPoint, pseudo_arclength!

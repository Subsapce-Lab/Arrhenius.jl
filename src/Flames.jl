abstract type AbstractPremixedFlame end

"A freely propagating, adiabatic, planar ideal-gas premixed flame."
mutable struct FreeFlame{G<:Solution} <: AbstractPremixedFlame
    gas::G
    grid::Vector{Float64}
    pressure::Float64
    inlet_temperature::Float64
    inlet_Y::Vector{Float64}
    state::Matrix{Float64}
    anchor::Int
    fixed_temperature::Float64
    dependent_species::Int
    converged::Bool
    transport_model::Symbol
    soret_enabled::Bool
    multicomponent_data::Union{Nothing,MultiTransportData}
    flux_gradient_basis::Symbol
end

"A planar premixed flame with prescribed inlet mass flux."
mutable struct BurnerFlame{G<:Solution} <: AbstractPremixedFlame
    gas::G
    grid::Vector{Float64}
    pressure::Float64
    inlet_temperature::Float64
    inlet_Y::Vector{Float64}
    state::Matrix{Float64}
    anchor::Int
    fixed_temperature::Float64
    dependent_species::Int
    converged::Bool
    mass_flux::Float64
    profile_positions::Vector{Float64}
    profile_temperatures::Vector{Float64}
    imposed_temperature::Vector{Float64}
    transport_model::Symbol
    soret_enabled::Bool
    multicomponent_data::Union{Nothing,MultiTransportData}
    flux_gradient_basis::Symbol
end

"""
    FreeFlame(gas; T=300, P=one_atm, X, width=0.03, grid=nothing)

Construct a native Julia premixed flame with mixture-averaged diffusion using
mole-fraction gradients. The inlet mass flux (and flame speed) is an eigenvalue.
Initialization uses native constant-enthalpy chemical equilibrium. Call `solve!`
to solve the species and energy equations with adaptive grid refinement.
"""
function FreeFlame(gas::Solution; T=300.0, P=one_atm, X, width=0.03, grid=nothing,
        transport_model=:mixture_averaged, multicomponent_data=nothing, soret=false,
        flux_gradient_basis=:mole)
    isfinite(T) && 200 <= T <= 6000 && isfinite(P) && P > 0 ||
        throw(ArgumentError("finite inlet temperature in 200–6000 K and positive pressure required"))
    gas.trans.poly_order == 5 || throw(ArgumentError("regenerate the sidecar to include native transport fits"))
    x = mole_fractions(gas,X)
    eq = equilibrate(gas; T, P, X=x, mode=:HP)
    eq.T > T+10 || throw(ArgumentError("the inlet does not support an exothermic adiabatic flame"))
    z = isnothing(grid) ? width .* [0,.2,.3,.35,.4,.5,.6,.8,1] : Float64.(grid)
    length(z) >= 5 && all(isfinite,z) && all(>(0),diff(z)) ||
        throw(ArgumentError("the grid needs at least five strictly increasing finite coordinates"))
    y = x .* gas.MW ./ dot(x,gas.MW)
    n = gas.n_species
    state = zeros(n+2,length(z))
    rho = P*dot(x,gas.MW)/(R*T)
    for j in eachindex(z)
        fraction = clamp(((z[j]-z[1])/(z[end]-z[1])-.3)/.2, 0, 1)
        state[1,j] = (T + fraction*(eq.T-T))/1000
        state[2:n+1,j] = y + fraction*(eq.Y-y)
        state[end,j] = rho
    end
    anchor = clamp(argmin(abs.(state[1,:] .- (.75*T+.25*eq.T)/1000)),2,length(z)-1)
    flame = FreeFlame(gas,z,Float64(P),Float64(T),y,state,anchor,
        1000*state[1,anchor],argmax(y),false,:mixture_averaged,false,nothing,:mole)
    return set_transport!(flame,transport_model; data=multicomponent_data,soret,flux_gradient_basis)
end

"""
    BurnerFlame(gas; mdot, T=300, P=one_atm, X, width=0.03, grid=nothing)

Construct a burner-stabilized premixed flame. `mdot` is the inlet mass flux in
kg/(m² s). The energy equation is enabled unless `set_temperature_profile!`
specifies a measured temperature profile.
"""
function BurnerFlame(gas::Solution; mdot,T=300.0,P=one_atm,X,width=.03,grid=nothing,
        transport_model=:mixture_averaged,multicomponent_data=nothing,soret=false,
        flux_gradient_basis=:mole)
    isfinite(mdot) && mdot > 0 || throw(ArgumentError("mass flux must be finite and positive"))
    grid = isnothing(grid) ? width .* [0,.1,.2,.3,.5,.7,1] : grid
    base = FreeFlame(gas; T,P,X,width,grid,transport_model,multicomponent_data,soret,flux_gradient_basis)
    Teq = base.state[1,end]
    Yeq = copy(base.state[2:end-1,end])
    for j in eachindex(base.grid)
        fraction = clamp(5*(base.grid[j]-base.grid[1])/(base.grid[end]-base.grid[1]),0,1)
        base.state[1,j] = T/1000+fraction*(Teq-T/1000)
        base.state[2:end-1,j] = base.inlet_Y+fraction*(Yeq-base.inlet_Y)
        base.state[end,j] = mdot
    end
    return BurnerFlame(base.gas,base.grid,base.pressure,base.inlet_temperature,
        base.inlet_Y,base.state,base.anchor,base.fixed_temperature,
        base.dependent_species,false,Float64(mdot),Float64[],Float64[],Float64[],
        base.transport_model,base.soret_enabled,base.multicomponent_data,base.flux_gradient_basis)
end

"""
    set_transport!(flame, model; data=flame.multicomponent_data, soret=false)

Select `:mixture_averaged` or `:multicomponent` diffusion. Multicomponent
diffusion and either model's Soret diffusion require `MultiTransportData`
exported for the same mechanism. `soret=true` includes thermal diffusion.
Use `flux_gradient_basis=:mass` or `:mole` for mixture-averaged diffusion.
An existing solution is retained as the initial guess for the next `solve!`.
"""
function set_transport!(f::AbstractPremixedFlame,model;data=f.multicomponent_data,soret=false,
        flux_gradient_basis=f.flux_gradient_basis)
    model = Symbol(replace(String(model),"-"=>"_"))
    model in (:mixture_averaged,:multicomponent) || throw(ArgumentError("unknown flame transport model"))
    flux_gradient_basis = Symbol(flux_gradient_basis)
    flux_gradient_basis in (:mole,:mass) || throw(ArgumentError("flux gradient basis must be :mole or :mass"))
    soret isa Bool || throw(ArgumentError("soret must be true or false"))
    if model == :multicomponent || soret
        data isa MultiTransportData || throw(ArgumentError("multicomponent transport requires MultiTransportData"))
        data.species_names == f.gas.species_names && data.molecular_weights ≈ f.gas.MW ||
            throw(ArgumentError("multicomponent data must match the flame mechanism"))
    end
    f.transport_model,f.soret_enabled,f.multicomponent_data = model,soret,data
    f.flux_gradient_basis = flux_gradient_basis
    f.converged = false
    return f
end

function _interpolate_profile(positions,values,z)
    j = clamp(searchsortedlast(positions,z),1,length(positions)-1)
    fraction = (z-positions[j])/(positions[j+1]-positions[j])
    return (1-fraction)*values[j]+fraction*values[j+1]
end

"""
    set_temperature_profile!(flame::BurnerFlame, positions, temperatures; relative=true)

Prescribe temperature [K] using piecewise-linear interpolation. Relative positions
span 0–1; absolute positions are in meters. The profile must cover the domain and
match the burner temperature at its inlet. Grid refinement preserves this profile.
"""
function set_temperature_profile!(f::BurnerFlame,positions,temperatures;relative=true)
    z,T = Float64.(positions),Float64.(temperatures)
    length(z) == length(T) && length(z) >= 2 || throw(DimensionMismatch("at least two position/temperature pairs required"))
    all(isfinite,z) && all(>(0),diff(z)) && all(t->isfinite(t)&&200<=t<=6000,T) ||
        throw(ArgumentError("profile requires increasing positions and temperatures in 200–6000 K"))
    if relative
        z = f.grid[1] .+ z .* (f.grid[end]-f.grid[1])
    end
    z[1] <= f.grid[1] && z[end] >= f.grid[end] || throw(ArgumentError("profile must cover the entire domain"))
    isapprox(_interpolate_profile(z,T,f.grid[1]),f.inlet_temperature;atol=1e-6,rtol=0) ||
        throw(ArgumentError("profile inlet must match burner temperature"))
    f.profile_positions,f.profile_temperatures = z,T
    f.imposed_temperature = [_interpolate_profile(z,T,x) for x in f.grid]
    f.state[1,:] .= f.imposed_temperature ./ 1000
    f.converged = false
    return f
end

"Temperature [K] at each grid point."
temperature(f::AbstractPremixedFlame) = 1000 .* vec(f.state[1,:])
"Species mass fractions, with species along rows and grid points along columns."
mass_fractions(f::AbstractPremixedFlame) = copy(f.state[2:end-1,:])
"Axial velocity [m/s] at each grid point."
function velocity(f::AbstractPremixedFlame)
    return [f.state[end,j]*R*(1000*f.state[1,j])*
        sum(f.state[k+1,j]/f.gas.MW[k] for k in 1:f.gas.n_species)/f.pressure
        for j in eachindex(f.grid)]
end
flame_speed(f::AbstractPremixedFlame) = velocity(f)[1]

struct FlameWorkspace{K}
    kinetics::K
    transport::TransportWorkspace
    X::Matrix{Float64}
    rho::Vector{Float64}
    cp::Matrix{Float64}
    h::Matrix{Float64}
    source::Matrix{Float64}
    flux::Matrix{Float64}
    conductivity::Vector{Float64}
    diffusion_prefactor::Matrix{Float64}
    C::Vector{Float64}
    entropy::Vector{Float64}
    xmid::Vector{Float64}
    ymid::Vector{Float64}
    band::Matrix{Float64}
    perturbed::Matrix{Float64}
    residual_perturbed::Matrix{Float64}
    steps::Vector{Float64}
    multi_transport::Union{Nothing,MultiTransportWorkspace}
    multi_prefactor::Array{Float64,3}
    thermal_diffusion::Matrix{Float64}
    mixture_thermal::Union{Nothing,MixtureThermalDiffusionWorkspace}
    rate_caches::Vector{_KineticsTemperatureCache}
end
function FlameWorkspace(f::AbstractPremixedFlame)
    n, N = f.gas.n_species, length(f.grid)
    return FlameWorkspace(KineticsWorkspace(f.gas.reaction),TransportWorkspace(f.gas),
        zeros(n,N),zeros(N),zeros(n,N),zeros(n,N),zeros(n,N),zeros(n,N-1),zeros(N-1),
        zeros(n,N-1),zeros(n),zeros(n),zeros(n),zeros(n),
        zeros(6*(n+2)-2,(n+2)*N),zeros(n+2,N),zeros(n+2,N),zeros(N),
        f.transport_model == :multicomponent ? MultiTransportWorkspace(f.multicomponent_data) : nothing,
        zeros(n,n,f.transport_model == :multicomponent ? N-1 : 0),zeros(n,N-1),
        f.transport_model == :mixture_averaged && f.soret_enabled ?
            MixtureThermalDiffusionWorkspace(f.multicomponent_data) : nothing,
        [_KineticsTemperatureCache(f.gas.reaction) for _ in 1:N])
end

function _flame_properties!(w,f,u; update_transport=true,nodes=eachindex(f.grid))
    gas, n, N = f.gas, f.gas.n_species, length(f.grid)
    @inbounds for j in nodes
        T = 1000*u[1,j]
        sumY = 0.0
        inverseMW = 0.0
        for k in 1:n
            y = max(u[k+1,j],0.0)
            sumY += y
            inverseMW += y/gas.MW[k]
        end
        meanMW = sumY/inverseMW
        w.rho[j] = f.pressure*meanMW/(R*T)
        for k in 1:n
            w.X[k,j] = max(u[k+1,j],0.0)/gas.MW[k]/inverseMW
            w.C[k] = f.pressure/(R*T)*w.X[k,j]
        end
        x = @view w.X[:,j]
        h = @view w.h[:,j]
        cp = @view w.cp[:,j]
        cal_h_RT!(h,gas,T,f.pressure,x)
        cal_cp_R!(cp,gas,T,f.pressure,x)
        cal_s0_R!(w.entropy,gas,T,f.pressure,x)
        h .*= R*T
        cp .*= R
        w.entropy .*= R
        wdot!(@view(w.source[:,j]),gas.reaction,T,w.C,w.entropy,h,w.kinetics;
            temperature_cache=w.rate_caches[j])
    end
    @inbounds for j in 1:N-1
        Tmid = 500*(u[1,j]+u[1,j+1])
        ysum, inverseMW = 0.0, 0.0
        for k in 1:n
            w.ymid[k] = max(0.0, .5*(u[k+1,j]+u[k+1,j+1]))
            ysum += w.ymid[k]
            inverseMW += w.ymid[k]/gas.MW[k]
        end
        meanMW = ysum/inverseMW
        for k in 1:n
            w.ymid[k] /= ysum
            w.xmid[k] = w.ymid[k]*meanMW/gas.MW[k]
        end
        if update_transport
            if f.transport_model == :multicomponent
                w.conductivity[j] = multicomponent_transport!(w.multi_transport,
                    f.multicomponent_data,gas,f.pressure,Tmid,w.xmid)
                for k in 1:n, l in 1:n
                    w.multi_prefactor[k,l,j] = f.pressure/(R*Tmid*meanMW)*
                        gas.MW[k]*gas.MW[l]*w.multi_transport.diffusion[k,l]
                end
                w.thermal_diffusion[:,j] .= w.multi_transport.thermal_diffusion
            else
                _, w.conductivity[j] = mixture_transport!(w.transport,gas,f.pressure,Tmid,w.xmid;
                    basis=f.flux_gradient_basis)
                for k in 1:n
                    weight = f.flux_gradient_basis == :mass ? meanMW : gas.MW[k]
                    w.diffusion_prefactor[k,j] = f.pressure/(R*Tmid)*weight*w.transport.diffusion[k]
                end
                if f.soret_enabled
                    w.thermal_diffusion[:,j] .= mixture_thermal_diffusion!(w.mixture_thermal,
                        f.multicomponent_data,f.pressure,Tmid,w.xmid)
                end
            end
        end
        dz = f.grid[j+1]-f.grid[j]
        if f.transport_model == :multicomponent
            for k in 1:n
                flux = 0.0
                for l in 1:n
                    flux += w.multi_prefactor[k,l,j]*(w.X[l,j+1]-w.X[l,j])/dz
                end
                w.flux[k,j] = flux
            end
        else
            fluxsum = 0.0
            for k in 1:n
                gradient = f.flux_gradient_basis == :mass ?
                    (u[k+1,j+1]-u[k+1,j])/dz : (w.X[k,j+1]-w.X[k,j])/dz
                w.flux[k,j] = -w.diffusion_prefactor[k,j]*gradient
                fluxsum += w.flux[k,j]
            end
            for k in 1:n
                w.flux[k,j] -= u[k+1,j]*fluxsum
            end
        end
        if f.soret_enabled
            for k in 1:n
                w.flux[k,j] -= w.thermal_diffusion[k,j]*1000*(u[1,j+1]-u[1,j])/(dz*Tmid)
            end
        end
    end
end

const _flame_timescale = 1e-4

"Evaluate the discretized steady species, energy, and mass-flow residual."
function flame_residual!(residual, f::AbstractPremixedFlame, u=f.state, w=FlameWorkspace(f);
        previous=nothing, dt=Inf, update_transport=true,nodes=eachindex(f.grid))
    n,N = f.gas.n_species,length(f.grid)
    _flame_properties!(w,f,u; update_transport,nodes)
    z, MW = f.grid, f.gas.MW
    @inbounds for j in 1:N
        if f isa BurnerFlame
            residual[end,j] = j == 1 ? u[end,j]-f.mass_flux : u[end,j]-u[end,j-1]
        elseif j == f.anchor
            residual[end,j] = u[1,j] - f.fixed_temperature/1000
        elseif j < f.anchor
            residual[end,j] = u[end,j+1]-u[end,j]
        else
            residual[end,j] = u[end,j]-u[end,j-1]
        end
        if j == 1
            residual[1,j] = u[1,j]-f.inlet_temperature/1000
            for k in 1:n
                residual[k+1,j] = u[end,j]*(f.inlet_Y[k]-u[k+1,j])-w.flux[k,1]
            end
        elseif j == N
            residual[1,j] = u[1,j]-u[1,j-1]
            for k in 1:n
                residual[k+1,j] = u[k+1,j]-u[k+1,j-1]
            end
        else
            left = z[j]-z[j-1]
            right = z[j+1]-z[j]
            cell = .5*(left+right)
            mdot = u[end,j]
            # Positive axial flow: first-order upwind convection, centered diffusion.
            dTdz = 1000*(u[1,j]-u[1,j-1])/left
            cpmean, chemical, enthalpyflux = 0.0,0.0,0.0
            for k in 1:n
                cpmean += u[k+1,j]*w.cp[k,j]/MW[k]
                chemical += w.h[k,j]*w.source[k,j]
                enthalpyflux += .5*(w.flux[k,j-1]+w.flux[k,j])*
                    (w.h[k,j]-w.h[k,j-1])/(left*MW[k])
                residual[k+1,j] = _flame_timescale/w.rho[j]*(MW[k]*w.source[k,j] -
                    (w.flux[k,j]-w.flux[k,j-1])/cell - mdot*(u[k+1,j]-u[k+1,j-1])/left)
            end
            conduction = 1000*(w.conductivity[j]*(u[1,j+1]-u[1,j])/right -
                w.conductivity[j-1]*(u[1,j]-u[1,j-1])/left)/cell
            residual[1,j] = _flame_timescale/(1000*w.rho[j]*cpmean)*
                (conduction-chemical-enthalpyflux-mdot*cpmean*dTdz)
            if previous !== nothing
                for k in 1:n+1
                    residual[k,j] -= _flame_timescale/dt*(u[k,j]-previous[k,j])
                end
            end
        end
        if f isa BurnerFlame && !isempty(f.imposed_temperature)
            residual[1,j] = u[1,j]-f.imposed_temperature[j]/1000
        end
        residual[f.dependent_species+1,j] = sum(@view(u[2:n+1,j]))-1
    end
    return residual
end

# Three grid colors exploit the nearest-neighbor block stencil. Each residual
# evaluation perturbs one component at every third point without overlap.
function _flame_jacobian(f,u,w,r; previous=nothing,dt=Inf)
    B,N = size(u)
    band = w.band
    fill!(band,0)
    kl = ku = 2*B-1
    perturbed = w.perturbed
    perturbed .= u
    rp = w.residual_perturbed
    steps = w.steps
    base_properties = (copy(w.X),copy(w.rho),copy(w.cp),copy(w.h),copy(w.source))
    for k in 1:B, color in 1:3
        for j in color:3:N
            steps[j] = 1e-7*max(abs(u[k,j]), k == 1 ? .1 : 1e-5)
            perturbed[k,j] = u[k,j]+steps[j]
        end
        flame_residual!(rp,f,perturbed,w; previous,dt,update_transport=false,nodes=color:3:N)
        for j in color:3:N
            column = (j-1)*B+k
            for jj in max(1,j-1):min(N,j+1), kk in 1:B
                row = (jj-1)*B+kk
                band[kl+ku+1+row-column,column] = (rp[kk,jj]-r[kk,jj])/steps[j]
            end
            perturbed[k,j] = u[k,j]
        end
        w.X .= base_properties[1]
        w.rho .= base_properties[2]
        w.cp .= base_properties[3]
        w.h .= base_properties[4]
        w.source .= base_properties[5]
    end
    return band
end

function _flame_newton!(f,w; previous=nothing,dt=Inf,maxiters=35,tolerance=1e-8,loglevel=0)
    u = f.state
    r,trial,rt = similar(u),similar(u),similar(u)
    bandwidth = 2*size(u,1)-1
    pivots = LinearAlgebra.BlasInt[]
    age = 5
    last_contraction = Inf
    for iteration in 1:maxiters
        flame_residual!(r,f,u,w; previous,dt)
        residual_norm = norm(r,Inf)
        residual_norm < tolerance && return true
        refresh = age >= 5 || last_contraction > .7
        step = try
            if refresh
                J = _flame_jacobian(f,u,w,r; previous,dt)
                _,pivots = LinearAlgebra.LAPACK.gbtrf!(bandwidth,bandwidth,length(u),J)
                age = 0
            end
            correction = -vec(copy(r))
            LinearAlgebra.LAPACK.gbtrs!('N',bandwidth,bandwidth,length(u),w.band,pivots,correction)
            reshape(correction,size(u))
        catch e
            e isa SingularException || rethrow()
            return false
        end
        all(isfinite,step) || return false
        alpha = 1.0
        for j in axes(u,2), k in axes(u,1)
            # Small negative species values are permitted in Newton iterates;
            # thermodynamic and kinetic evaluations use the nonnegative state.
            low = k == 1 ? .2 : k == size(u,1) ? 1e-6 : -1e-7
            high = k == 1 ? 6.0 : k == size(u,1) ? 100.0 : 1.00001
            if step[k,j] < 0
                alpha = min(alpha,.99*(u[k,j]-low)/(-step[k,j]))
            elseif step[k,j] > 0
                alpha = min(alpha,.99*(high-u[k,j])/step[k,j])
            end
        end
        accepted = false
        for backtrack in 1:24
            @. trial = u + alpha*step
            flame_residual!(rt,f,trial,w; previous,dt)
            if all(isfinite,rt) && norm(rt) < norm(r)*(1-1e-4*alpha)
                u .= trial
                accepted = true
                break
            end
            alpha *= .5
        end
        loglevel > 1 && println("Newton ",iteration," residual=",residual_norm," damping=",alpha)
        if !accepted || alpha <= 1e-10
            refresh && return false
            age = 5
            continue
        end
        last_contraction = norm(rt)/norm(r)
        age += 1
    end
    return false
end

function _flame_steady!(f; loglevel=0,max_time_steps=500,timestep=Ref(1e-6))
    w = FlameWorkspace(f)
    _flame_newton!(f,w; loglevel) && return true
    dt = timestep[]
    for step in 1:max_time_steps
        previous = copy(f.state)
        if _flame_newton!(f,w; previous,dt,maxiters=20,loglevel=0)
            dt = min(dt*1.5,1.0)
            timestep[] = dt
            if step % 10 == 0
                loglevel > 0 && println("Transient step ",step," dt=",dt)
                _flame_newton!(f,w; loglevel) && return true
            end
        else
            f.state .= previous
            dt *= .25
            timestep[] = dt
            dt > 1e-12 || return false
        end
    end
    return false
end

function _refine_flame!(f; ratio=3.,slope=.06,curve=.12,max_points=1000)
    u,z = f.state,f.grid
    B,N = size(u)
    insert = falses(N-1)
    spacing = diff(z)
    for k in 1:B
        # An imposed piecewise-linear profile has slope discontinuities that
        # cannot be removed by refinement. Its shape is supplied, not solved.
        if k in (1,B) && f isa BurnerFlame && !isempty(f.imposed_temperature)
            continue
        end
        # Refine physical velocity rather than the mass-flux algebraic variable.
        values = k == B ? velocity(f) : @view(u[k,:])
        low,high = extrema(values)
        span = high-low
        threshold = k == 1 ? sqrt(eps(Float64))/1000 : sqrt(eps(Float64))
        if span > .01*max(abs(low),abs(high))
            for j in 1:N-1
                abs(values[j+1]-values[j]) > slope*span+threshold &&
                    spacing[j]>=2e-10 && (insert[j] = true)
            end
        end
        gradients = diff(values) ./ spacing
        glow,ghigh = extrema(gradients)
        gspan = ghigh-glow
        if gspan > .01*max(abs(glow),abs(ghigh))
            for j in 1:N-2
                if abs(gradients[j+1]-gradients[j]) > curve*gspan+threshold/spacing[j] &&
                        min(spacing[j],spacing[j+1])>=2e-10
                    insert[j] = true
                    insert[j+1] = true
                end
            end
        end
    end
    for j in 1:N-2
        left,right = z[j+1]-z[j],z[j+2]-z[j+1]
        left > ratio*right && (insert[j] = true)
        right > ratio*left && (insert[j+1] = true)
    end
    count(insert) == 0 && return false
    N+count(insert) <= max_points || error("flame refinement exceeds max_points=$max_points")
    newz = Float64[]
    columns = Vector{Float64}[]
    anchor_z = z[f.anchor]
    for j in 1:N
        push!(newz,z[j]); push!(columns,u[:,j])
        if j < N && insert[j]
            push!(newz,(z[j]+z[j+1])/2)
            push!(columns,(u[:,j]+u[:,j+1])/2)
        end
    end
    f.grid = newz
    f.state = reduce(hcat,columns)
    f.anchor = findfirst(==(anchor_z),newz)
    if f isa BurnerFlame && !isempty(f.imposed_temperature)
        f.imposed_temperature = [_interpolate_profile(f.profile_positions,f.profile_temperatures,x) for x in newz]
        f.state[1,:] .= f.imposed_temperature ./ 1000
    end
    return true
end

"""
    solve!(flame; refine_grid=true, ratio=3, slope=0.06, curve=0.12,
           max_points=1000, max_time_steps=500, loglevel=0)

Solve a native premixed flame using damped Newton iterations and implicit
pseudo-time stepping. Failure raises an error and leaves `converged=false`.
"""
function solve!(f::AbstractPremixedFlame; refine_grid=true,ratio=3.,slope=.06,curve=.12,
        max_points=1000,max_time_steps=500,loglevel=0,
        initial_time_step=f isa FreeFlame ? 1e-6 : 1e-5)
    isfinite(ratio) && ratio > 1 && 0 < slope <= 1 && 0 < curve <= 1 || throw(ArgumentError("invalid refinement criteria"))
    isfinite(initial_time_step) && initial_time_step > 0 || throw(ArgumentError("initial time step must be finite and positive"))
    f.converged = false
    timestep = Ref(Float64(initial_time_step))
    for pass in 1:40
        loglevel > 0 && println("Solving on ",length(f.grid)," points")
        _flame_steady!(f; loglevel,max_time_steps,timestep) || error("flame solve failed on $(length(f.grid)) points")
        if !refine_grid || !_refine_flame!(f;ratio,slope,curve,max_points)
            f.converged = true
            return f
        end
    end
    error("flame refinement did not converge after 40 passes")
end
export FreeFlame, BurnerFlame, set_transport!, set_temperature_profile!, solve!, temperature, mass_fractions, velocity, flame_speed, flame_residual!

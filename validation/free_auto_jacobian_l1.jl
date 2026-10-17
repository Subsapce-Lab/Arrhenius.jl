using Arrhenius, LinearAlgebra

let
length(ARGS)==1 || error("usage: julia validation/free_auto_jacobian_l1.jl output.npz")
ref=Arrhenius.npzread(joinpath(@__DIR__,"..","test","data","cantera","free_auto_stagni.npz"))
mechanism=joinpath(@__DIR__,"..","mechanism","stagni_2023.yaml")
gas=CreateSolution(mechanism); grid=vec(ref["grid"])
f=FreeFlame(gas;T=300.0,P=one_atm,X="NH3:0.7,H2:0.3,O2:0.5625,N2:2.115",
    grid,width=grid[end]-grid[1],transport_model=:mixture_averaged,discretization=:finite_difference)
n,N=gas.n_species,length(grid)
f.state[1,:].=vec(ref["values/T"])./1000
for (k,name) in enumerate(gas.species_names)
    f.state[k+1,:].=vec(ref["values/"*name])
end
for j in 1:N
    rho=Arrhenius._flame_density(f,1000f.state[1,j],@view(f.state[2:n+1,j])).density
    f.state[end,j]=rho*ref["values/velocity"][j]
end
w=Arrhenius.FlameWorkspace(f); r=similar(f.state)
previous=copy(f.state); dt=1e-5
flame_residual!(r,f,f.state,w;previous,dt)
band=Arrhenius._flame_jacobian(f,f.state,w,r;previous,dt,analytic=true)
B=size(f.state,1); bw=2B-1; dense=zeros(N*B,N*B)
for col in 1:N*B, row in max(1,col-bw):min(N*B,col+bw)
    dense[row,col]=band[2bw+1+row-col,col]
end
step=permutedims(reshape(dense\(-vec(permutedims(r))),N,B))
alpha=Inf; limiter=(0,0)
for j in 1:N,k in 1:B
    lo,hi=Arrhenius._flame_bounds(f,k,B); x=f.state[k,j]; d=step[k,j]
    a=d<0 ? (x-lo)/(-d) : d>0 ? (hi-x)/d : Inf
    if a<alpha; alpha=a; limiter=(j,k); end
end
println("alpha=",alpha," limiter=",limiter," norm_step=",norm(step))
Arrhenius.npzwrite(ARGS[1],Dict("state"=>f.state,"grid"=>grid,"residual"=>r,"previous"=>previous,"dt"=>[dt],"jacobian_band"=>band,"newton_step"=>step))
end

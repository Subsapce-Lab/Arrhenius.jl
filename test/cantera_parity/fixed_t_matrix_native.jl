# Export Arrhenius residual and banded Jacobian at a stored Cantera fixed-T state.
# Usage: julia --project=<Arrhenius> fixed_t_matrix_native.jl spec.json state.npz out.npz
using Arrhenius, YAML, NPZ, LinearAlgebra
BLAS.set_num_threads(1)
function run_case(specpath,statepath,outfile)
spec=YAML.load_file(specpath); d=npzread(statepath); out=outfile
gas=CreateSolution(spec["mechanism"]*".yaml")
z=vec(d["grid"]); T=vec(d["T"]); Y=Matrix{Float64}(d["Y"]); U=vec(d["velocity"])
X=Dict(String(k)=>parse(Float64,v) for (k,v) in split.(split(spec["X"],","),":"))
transport=spec["transport"]=="multicomponent" ? :multicomponent : :mixture_averaged
data=transport==:multicomponent ? MultiTransportData(spec["mechanism"]*".yaml.multicomponent.npz",gas;mechanism=spec["mechanism"]*".yaml") : nothing
f=BurnerFlame(gas;mdot=spec["mdot"],T=spec["T"],P=spec["P_atm"]*one_atm,X,
    width=spec["width"],transport_model=transport,multicomponent_data=data,
    discretization=:finite_difference,grid=z)
set_temperature_profile!(f,Float64.(spec["profile_z"]),Float64.(spec["profile_T"]);relative=false)
length(f.grid)==length(z) || error("profile insertion changed stored grid")
n=gas.n_species
meanMW=1 ./ (transpose(Y)*(1 ./ gas.MW))
rho=spec["P_atm"]*one_atm .* meanMW ./ (Arrhenius.R .* T)
f.state[1,:]=T./1000
f.state[2:n+1,:]=Y
f.state[end,:]=rho.*U
r=similar(f.state); w=Arrhenius.FlameWorkspace(f)
Arrhenius.flame_residual!(r,f,f.state,w)
J=Arrhenius._flame_jacobian(f,f.state,w,r;analytic=true)
B,N=size(f.state); m=B*N; bw=2B-1
bands=zeros(2bw+1,m)
for col in 1:m
    for row in max(1,col-bw):min(m,col+bw)
        bands[bw+1+row-col,col]=J[2bw+1+row-col,col]
    end
end
# Convert native variables (T/1000,Y,mdot) to Cantera variables (T,Y,U).
scales=repeat(vcat(0.001,ones(n),rho),N)
bands .*= reshape(scales,1,:)
npzwrite(out,Dict("r_arr"=>vec(r),"bands_arr"=>bands,"bw"=>[bw],"rho"=>rho))
println("case=$(spec["id"]) exported points=$N dofs=$m band=$bw residual_inf=$(norm(r,Inf)) file=$out")

end

length(ARGS)%3==0 && !isempty(ARGS) || error("provide spec/state/output triples")
for i in 1:3:length(ARGS)
    run_case(ARGS[i],ARGS[i+1],ARGS[i+2])
end

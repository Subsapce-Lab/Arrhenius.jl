# Usage: julia --project=SOLVER_ENV example/reactors/non_ideal_shock_tube.jl MECHANISM_DIR [--sweep]
# The caller-owned solver environment needs SciMLBase and OrdinaryDiffEqSDIRK.
# MECHANISM_DIR contains dodecane_IG.yaml, its native kinetic sidecar, and
# dodecane_RK.yaml. validation/real_gas_reactor_cases.py prepares the paired input
# files and independent Cantera references. No Cantera call occurs in this file.
using Arrhenius, LinearAlgebra, Printf
include(joinpath(@__DIR__,"real_gas_ode_solver.jl"))
BLAS.set_num_threads(1)

function shocktube_ignition(reactor;end_time=.005,save_stride=20)
    solution = solve_reactor(reactor,(0.,end_time);integrator=shocktube_sdirk,
        reltol=1e-13,abstol=1e-26,save_everystep=true,dense=false,maxiters=1_000_000)
    # Match the source's every-20th-accepted-step sampling convention. Adaptive
    # step times differ between solvers, so sampled peak times can differ slightly.
    indices = collect((save_stride+1):save_stride:length(solution.t))
    isempty(indices) && throw(ErrorException("too few integration steps for the requested sampling stride"))
    oh = findfirst(==("oh"),reactor.gas.species_names)
    peak = argmax([solution.u[i][oh] for i in indices])
    delay = solution.t[indices[peak]]
    return (;delay,times=solution.t[indices],states=reduce(hcat,solution.u[indices]),solution)
end

function non_ideal_shock_tube(directory;sweep=false)
    gas = CreateSolution(joinpath(directory,"dodecane_IG.yaml"))
    model = RedlichKwongThermo(joinpath(directory,"dodecane_RK.yaml"))
    temperatures = sweep ? [1000,1250,1170,1120,1080,1040,1010,990,970,950,930,910,880,850,820,790,760] : [1000]
    composition = Dict("c12h26"=>1.,"o2"=>18.5,"n2"=>69.56)
    results = []
    for temperature in temperatures
        rk = RedlichKwongReactor(gas,model;temperature,pressure=40one_atm,mole_fractions=composition)
        ig = IdealGasReactor(gas;temperature,pressure=40one_atm,mole_fractions=composition,constraint=:constant_volume)
        real = shocktube_ignition(rk)
        ideal = shocktube_ignition(ig)
        @printf("T=%4d K: RK %.9g s, ideal %.9g s; ideal-gas error %.4g%%\n",
            temperature,real.delay,ideal.delay,100*(ideal.delay-real.delay)/real.delay)
        push!(results,(;temperature,real,ideal))
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    1 <= length(ARGS) <= 2 || error("supply the prepared mechanism directory and optional --sweep")
    non_ideal_shock_tube(ARGS[1];sweep="--sweep" in ARGS)
end

# Optional arguments: prepared gri30.yaml and air.yaml, each with its .npz sidecar.
# Uses GRI species for both streams when no separate air mechanism is supplied.
using Arrhenius
include("network_cases.jl")
include("network_solver.jl")

gas = CreateSolution(isempty(ARGS) ? joinpath(@__DIR__,"..","..","mechanism","gri30.yaml") : ARGS[1])
air = length(ARGS) >= 2 ? CreateSolution(ARGS[2]) : gas
network = mixing_network(gas,air)
result = solve_network_steady(network; integrator=native_network_bdf, interval=0.2,
    max_time=5.0, steady_tolerance=1e-8, reltol=1e-9, abstol=1e-15,
    save_everystep=false,save_start=false)
mixed = network_diagnostics(network,result.state,result.time).nodes.mixer
println((temperature_K=mixed.temperature,pressure_Pa=mixed.pressure,
         mass_fractions=Dict(zip(gas.species_names,mixed.mass_fractions)), residual=result.residual))

# Native low-pressure burner with an energy equation or a prescribed temperature.
# Run: julia --project=. example/flames/burner_flame.jl [mechanism.yaml] [output-directory]
using Arrhenius

mechanism = isempty(ARGS) ? joinpath(@__DIR__,"..","..","mechanism","h2o2.yaml") : ARGS[1]
output = length(ARGS)>1 ? ARGS[2] : mktempdir()
mkpath(output)
gas = CreateSolution(mechanism)
flame = BurnerFlame(gas;T=373.,P=.05one_atm,X="H2:1.5,O2:1,AR:7",width=.5,mdot=.06)
solve!(flame;slope=.05,curve=.1)
println("Peak temperature: ",maximum(temperature(flame))," K")
save_flame(joinpath(output,"burner.csv"),flame)
save_flame(joinpath(output,"burner.npz"),flame)

# Selected demonstration input; replace this profile with measured temperatures.
set_temperature_profile!(flame,[0.,.005,.01,.02,.05,.1,1.],
    [373.,650.,1000.,1350.,1650.,1750.,1750.])
solve!(flame;slope=.05,curve=.1)
save_flame(joinpath(output,"burner-prescribed.csv"),flame)
println("Results: ",output)

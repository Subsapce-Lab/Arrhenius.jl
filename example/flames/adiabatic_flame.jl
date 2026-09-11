# Native hydrogen/oxygen/argon premixed flames with different transport models.
# Run: julia --project=. example/flames/adiabatic_flame.jl [mechanism.yaml] [output-directory]
using Arrhenius

mechanism = isempty(ARGS) ? joinpath(@__DIR__,"..","..","mechanism","h2o2.yaml") : ARGS[1]
output = length(ARGS)>1 ? ARGS[2] : mktempdir()
mkpath(output)
gas = CreateSolution(mechanism)
flame = FreeFlame(gas;T=300.,P=one_atm,X="H2:1.1,O2:1,AR:5",width=.03,
    flux_gradient_basis=:mass)
solve!(flame;slope=.02,curve=.04)
println("Mixture-averaged speed: ",flame_speed(flame)," m/s")
data = MultiTransportData(mechanism*".multicomponent.npz",gas;mechanism)
for (model,soret,label) in ((:mixture_averaged,true,"mixture-soret"),
        (:multicomponent,false,"multicomponent"),(:multicomponent,true,"multicomponent-soret"))
    set_transport!(flame,model;data,soret)
    solve!(flame;slope=.02,curve=.04)
    println(label," speed: ",flame_speed(flame)," m/s")
    save_flame(joinpath(output,label*".npz"),flame)
end
save_flame(joinpath(output,"adiabatic-flame.csv"),flame;basis=:mole)
println("Results: ",output)

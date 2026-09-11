using Arrhenius

gas = CreateSolution(joinpath(@__DIR__,"..","..","mechanism","gri30.yaml"))
fuel, oxidizer = "CH4", "O2:0.21,N2:0.79"
X = set_equivalence_ratio(gas,1.0;fuel,oxidizer)
println("Fresh mixture phi = ",equivalence_ratio(gas,X;fuel,oxidizer))
println("Fuel mass fraction = ",mixture_fraction(gas,X;fuel,oxidizer))

burnt = equilibrate(gas;T=300.0,P=one_atm,X,mode=:HP)
println("Adiabatic equilibrium T = ",burnt.T," K")
println("Burnt mixture phi = ",equivalence_ratio(gas,burnt.X;fuel,oxidizer))
println("Burnt mixture fraction = ",mixture_fraction(gas,burnt.X;fuel,oxidizer))

diluted = set_equivalence_ratio(gas,2.0;fuel="H2",oxidizer="O2",
    diluent="CO2:0.5,H2O:0.5",basis=:mass,fraction=(fuel=0.1,))
println("Diluted mixture phi (H2/O2 only) = ",equivalence_ratio(gas,diluted;
    fuel="H2",oxidizer="O2",include_species=["H2","O2"]))

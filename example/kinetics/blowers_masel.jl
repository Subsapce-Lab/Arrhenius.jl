using Arrhenius

# The reaction equations and parameters in Cantera's blowers_masel.py example.
gas = CreateSolution(joinpath(@__DIR__,"..","..","mechanism","gri30.yaml"))
rate = BlowersMaselRate(3.87e1,2.7,6260*1000*4.184,1e9)
index = Dict(name=>species_index(gas,name) for name in ("H","O","H2","OH","CH4","CH3"))
function reaction_enthalpies(gas,T,index)
    x = mole_fractions(gas,"H2:1")
    h = cal_h_RT(gas,T,one_atm,x).*(R*T)
    return (h[index["H"]]+h[index["OH"]]-h[index["O"]]-h[index["H2"]],
            h[index["CH3"]]+h[index["H2"]]-h[index["H"]]-h[index["CH4"]])
end
temperatures = collect(300.:100.:3400.)
rates = zeros(3,length(temperatures))
for (j,T) in enumerate(temperatures)
    h2,methane = reaction_enthalpies(gas,T,index)
    rates[:,j] = [rate.A*T^rate.b*exp(-rate.Ea0/(R*T)),
                  rate_constant(rate,T,h2),rate_constant(rate,T,methane)]
end
println("Forward rate constants at 300 K: ",rates[:,1])
effective_barrier = activation_energy(rate,first(reaction_enthalpies(gas,last(temperatures),index)))
enthalpies = collect(range(-5effective_barrier,5effective_barrier;length=100))
barriers = activation_energy.(Ref(rate),enthalpies)
println("Calculated ",length(temperatures)," temperature states and ",length(enthalpies)," enthalpy shifts.")

using Arrhenius, LinearAlgebra

# Porous-media example from Cantera. Pass the mechanism and matching
# multicomponent sidecar produced by mechanism/export_multicomponent.py.
# https://cantera.org/dev/examples/python/transport/dusty_gas.html
length(ARGS)==2 || error("usage: julia --project example/transport/dusty_gas.jl mechanism.yaml multicomponent.npz")
gas=CreateSolution(ARGS[1])
data=MultiTransportData(ARGS[2],gas)
g=DustyGasTransport(gas;porosity=.2,tortuosity=4.,mean_pore_radius=1.5e-7,
    mean_particle_diameter=1.5e-6,multicomponent_data=data)
composition=Dict("OH"=>1.,"H"=>2.,"O2"=>3.,"O"=>1e-8,"H2"=>1e-8,
    "H2O"=>1e-8,"H2O2"=>1e-8,"HO2"=>1e-8,"AR"=>1e-8)
X=mole_fractions(gas,composition)
T=500.; P=one_atm
display(dusty_gas_diffusion!(g,P,T,X))
println(dusty_gas_thermal_conductivity(g,P,T,X))
Y=X.*gas.MW./dot(X,gas.MW)
rho1=P*dot(X,gas.MW)/(R*T)
rho2=1.2rho1
println(dusty_gas_molar_fluxes(g,T,T,rho1,rho1,Y,Y,.001))
println(dusty_gas_molar_fluxes(g,T,T,rho1,rho2,Y,Y,.001))

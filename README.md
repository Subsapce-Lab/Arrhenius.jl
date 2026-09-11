# Arrhenius

We are in an early-development. Expect some adventures and rough edges.

## Installation

> pkg> add https://github.com/Subsapce-Lab/Arrhenius.jl

## Mechanism preprocessing

`CreateSolution` reads a Cantera YAML file and a same-name `.yaml.npz`
sidecar. Generate a sidecar using Cantera 3.2 or 4.0 dev and NumPy:

```bash
python mechanism/export_sidecar.py path/to/mechanism.yaml
```

The exporter supports elementary and three-body Arrhenius reactions,
Lindemann and Troe falloff reactions, pressure-dependent Arrhenius (PLOG)
reactions, Blowers–Masel rates and explicit reaction orders. Unsupported Cantera rate models are
rejected during preprocessing. Native ideal-gas thermochemistry supports
NASA7, multi-region NASA9, Shomate and constant-cp species models.

## Native calculations

The Julia solvers provide ideal-gas equilibrium at TP, TV, HP, UV, SP and SV;
isentropic states, frozen or equilibrium sound speeds, and constant-pressure stream mixing;
closed constant-pressure and constant-volume reactors; connected stirred
reactors with flow devices and heat-transfer walls; planar premixed free flames
and burner-stabilized flames; counterflow diffusion, opposed premixed and twin
premixed flames; and premixed impinging jets with an inert wall.
Gas/solid TP and HP equilibrium supports one initially absent, fixed-stoichiometry
condensed phase with NASA7 or constant-cp thermodynamics, including graphite formation.
Flame calculations support adaptive
grids, mixture-averaged and multicomponent diffusion, Soret diffusion, and
prescribed burner temperature profiles. Counterflow diffusion flames also
support optically thin CO₂/H₂O radiation. A native pure-water model provides
liquid/vapor states, saturation properties and Rankine-cycle calculations.
`critical_properties` returns critical temperature, pressure, density and
compressibility for eight TPX pure-fluid models.
Redlich–Kwong mixtures support gas and liquid cubic roots, caloric properties,
fugacity coefficients and partial molar properties.
Native Redlich–Kwong reaction rates and constant-volume adiabatic reactors
support nonideal shock-tube ignition calculations.
Reactor networks support prescribed, pressure-driven and inertial pistons,
including changing volumes, pressure work and wall heat transfer.
Ideal-surface chemistry supports elementary and sticking reactions, coverage
dependencies, fixed-stoichiometry solids and isothermal catalytic reactors.
`CoverageThermoModel` evaluates coverage-dependent standard enthalpy, entropy,
heat capacity and Gibbs energy with linear, polynomial, piecewise-linear or
interpolated self and cross interactions.
Isothermal catalytic plug flow supports a direct spatial DAE and a chain of
stirred reactors with species and elemental-flux diagnostics.
Catalytic impinging flames couple gas transport and chemistry to steady surface
coverages at a prescribed wall temperature.
Porous-media transport includes molecular and Knudsen diffusion and Darcy flow
through the native `DustyGasTransport` model.
Chemistry, thermodynamics, transport
evaluation and equation solves run in Julia. Cantera is used to preprocess
mechanisms and generate independent validation data.

```julia
using Arrhenius

gas = CreateSolution("mechanism/h2o2.yaml")
flame = FreeFlame(gas; T=300., P=one_atm, X="H2:1.1,O2:1,AR:5", width=.03)
solve!(flame)
println(flame_speed(flame))  # m/s
save_flame("flame.csv", flame; basis=:mole)
save_flame("flame.npz", flame)
```

For multicomponent or Soret diffusion, also export collision-integral data:

```bash
python mechanism/export_multicomponent.py mechanism/h2o2.yaml mechanism/h2o2.yaml.multicomponent.npz
```

```julia
data = MultiTransportData("mechanism/h2o2.yaml.multicomponent.npz", gas)
set_transport!(flame, :multicomponent; data, soret=true)
solve!(flame; slope=.02, curve=.04)
```

See [premixed flames](example/flames/adiabatic_flame.jl),
[burner flames](example/flames/burner_flame.jl),
[counterflow diffusion flames](example/flames/counterflow_diffusion.jl),
[opposed premixed flames](example/flames/counterflow_premixed.jl),
[twin flames](example/flames/counterflow_twin.jl),
[inert-wall flames](example/flames/counterflow_stagnation.jl),
[closed reactors](example/reactors), and
[thermodynamics](example/thermodynamics) for runnable calculations.
The [nozzle example](example/thermodynamics/isentropic.jl) computes adiabatic
area–Mach curves, and the [sound-speed example](example/thermodynamics/sound_speed.jl)
compares frozen and equilibrium acoustic responses.
The [mixing example](example/thermodynamics/mixing.jl) conserves species and
enthalpy while combining streams, then evaluates the mixture at chemical equilibrium.
Reactor examples use a caller-supplied Julia ODE integrator.

The [engine example](example/reactors/ic_engine.jl) computes eight revolutions
of an n-dodecane engine with prescribed injection, valves and piston motion.
It uses QNDF from OrdinaryDiffEqBDF, with SciMLBase and ForwardDiff in the
caller's Julia environment. Prepare the mechanism and run:

```bash
python validation/ic_engine_case.py engine-input
julia --project=ENGINE_ENV example/reactors/ic_engine.jl engine-input/dodecane_IG.yaml engine.csv
```

The CSV contains crank-angle profiles; the companion TOML contains heat,
pressure work, efficiency and CO estimates integrated over accepted states.
The callable entry `solve_ic_engine_qndf` is provided by
[NativeEngineQNDF](example/reactors/ic_engine_qndf_solver.jl).
The complete calculation has been checked against Cantera 4.0 on WSL;
[validation results and reproduction commands](validation/results/cantera4_wsl_ic_engine.json)
cover the species and thermal histories, conservation, and integrated outputs.

The [parallel transport example](example/transport/multiprocessing_viscosity.jl)
computes multicomponent thermal conductivity and viscosity over 5,000 temperatures
for a methane/oxygen/nitrogen mixture. It runs both serial and parallel sweeps,
using independent phase and transport storage for each Julia task:

```bash
python mechanism/export_sidecar.py path/to/gri30.yaml
python mechanism/export_multicomponent.py path/to/gri30.yaml path/to/gri30.yaml.multicomponent.npz
julia --threads=4 --project=. example/transport/multiprocessing_viscosity.jl path/to/gri30.yaml path/to/gri30.yaml.multicomponent.npz
```

The [gas/graphite example](example/thermodynamics/adiabatic.jl) computes adiabatic
equilibrium temperature and all gas/solid species amounts across 50 fuel/air mixtures.
Prepare the gas sidecar and condensed-phase parameters, then run:

```bash
python mechanism/export_condensed_phase.py graphite.yaml graphite --output graphite.condensed.json
julia --project=. example/thermodynamics/adiabatic.jl path/to/gri30.yaml graphite.condensed.json
```

The [CO2 equation-of-state example](example/thermodynamics/equations_of_state.jl)
computes ideal-gas, Redlich–Kwong and Span–Wagner properties over 1–100 bar at
300 K, including the stable vapor/liquid transition. It uses the native Julia
package Clapeyron 0.6.28 in a caller-supplied environment. Prepare Cantera's
`example_data/co2-thermo.yaml` and its sidecar, then export the full Helmholtz
parameters with CoolProp and run the Julia calculation:

```bash
julia --project=EOS_ENV -e 'using Pkg; Pkg.develop(path="."); Pkg.add(PackageSpec(name="Clapeyron", version="0.6.28"))'
python mechanism/export_sidecar.py path/to/co2-thermo.yaml
python mechanism/export_helmholtz.py CO2 carbon-dioxide.json
julia --project=EOS_ENV example/thermodynamics/equations_of_state.jl path/to/co2-thermo.yaml carbon-dioxide.json
```

For catalytic calculations, prepare an ideal-surface parameter archive:

```bash
python mechanism/export_surface.py diamond.yaml diamond_100 --output diamond.surface.npz
```

The [diamond-growth example](example/reactors/diamond_cvd.jl) uses native
surface rates and coverage integration. The [Blowers–Masel example](example/kinetics/blowers_masel.jl)
evaluates reaction rates and activation energies as temperature and enthalpy change.


## Publication

+ [Arrhenius.jl: A Differentiable Combustion Simulation Package](https://arxiv.org/pdf/2107.06172.pdf): overview of Arrhenius.jl and applications in deep mechanism reduction, uncertainty quantification, mechanism tuning and model discovery. [Slides in NCM21](https://www.slideshare.net/WeiqiJi/arrheniusjl-a-differentiable-combustion-simulation-package-248457895), [Vedio for NCM21](https://www.youtube.com/watch?v=X1mwpW78NvA).
+ [Machine Learning Approaches to Learn HyChem Models](https://www.researchgate.net/publication/350890609_Machine_Learning_Approaches_to_Learn_HyChem_Models): demonstrate 1000 times faster than genetic algorithms using commercial software for optimizing complex kinetic models.
+ [Neural Differential Equations for Inverse Modeling in Model Combustors](https://www.researchgate.net/publication/351223124_Neural_Differential_Equations_for_Inverse_Modeling_in_Model_Combustors)
+ [SGD-based Optimization in Modeling Combustion Kinetics: Case Studies in Tuning Mechanistic and Hybrid Kinetic Models](https://doi.org/10.1016/j.fuel.2022.124560)




## Applications

+ **Sensitivity analysis for auto-ignition** | [repo](https://github.com/DENG-MIT/ArrheniusActiveSubspace) | Features: auto-differentiation, multi-threading, sensitivity to all of three Arrhenius params A, b and Ea, active subspace based uncertainty quantification
+ **Sensitivity analysis for one-dimensional flames** | [repo](https://github.com/DENG-MIT/Arrhenius_Flame_1D) | Features: auto-differentiation, multi-threading, sensitivity to all of three Arrhenius params A, b and Ea.
+ **Automonous learn kinetic mechanism using neural network** | [repo](https://github.com/DENG-MIT/CRNN_HyChem) | Features: Chemical Reaction Neural Network (CRNN), Neural Ordinary Differential Equations.
+ **Deep Reduction** | [repo](https://github.com/DENG-MIT/DeepReduction) | Features: Two-stages mechanism reduction with deep learning.

**Examples**

> Note that some of the examples are in development and you can have early access by contacting [Weiqi Ji](mailto:weiqiji@mit.edu)
  + [Pyrolysis of JP10](./example/pyrolysis/pyrolysis.ipynb)
  + [Perfect Stirred Reactor](./example/perfect_stirred_reactor)
  + [Auto-ignition](https://github.com/DENG-MIT/NN-Ignition)
  + [Compute Jacobian using AD](https://gist.github.com/jiweiqi/21b8d149bd95b97d9ae948ab92e446df)

## Relevent packages
+ [ReactionMechanismSimulator.jl](https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl) The amazing Reaction Mechanism Simulator for simulating large chemical kinetic mechanisms
+ [Cantera](https://cantera.org/) A comprehensive C++ based combustion simulation package and with great python interface. Arrhenius relies on Cantera when it is applicable.

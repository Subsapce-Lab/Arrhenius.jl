# Boltzmann plasma thermochemistry

`PlasmaMechanism` reads the gas and charged-species NASA7 data, imported reaction sections, and collision tables of a `Boltzmann-two-term` YAML phase. Supported rates include reversible Arrhenius reactions, simple third bodies, Lindemann/Troe falloff, and irreversible two-temperature, Chebyshev, and electron-collision rates.

The gas temperature, stored electron temperature, electric field, and electron distribution are distinct state variables. Setting E/N stores the absolute field using the current density. `update_eedf!` updates the distribution and mobility while retaining the stored electron temperature. Gas-state and enthalpy setters retain the cached field, distribution, and mobility.

```julia
using Arrhenius
m = PlasmaMechanism("methane-plasma-pavan-2023.yaml";
    data_paths=["/path/to/thermodynamic-data"])
s = PlasmaState(m; temperature=300.0, pressure=101325.0,
    mole_fractions=Dict("CH4"=>0.095, "O2"=>0.19, "N2"=>0.715, "e"=>1e-11))
set_reduced_electric_field!(s, 190e-21 * exp(-32))
update_eedf!(s)
set_reduced_electric_field!(s, 190e-21)
update_eedf!(s)
rates = plasma_rates(s)
h = plasma_thermodynamics(s).h_mass
set_plasma_state!(s; temperature=330.0)
set_plasma_enthalpy!(s, h; pressure=101325.0)
```

The [state validation results](results/cantera4_wsl_pulse_thermochemistry.json) compare the methane pulse mechanism's 71 species and 440 reactions at five states against Cantera 4.0.0a2. They cover gas and electron thermodynamics, all rate families, density, field/EEDF caching, and enthalpy inversion. Tiny net production rates are also checked against high-precision sums of independently parsed net stoichiometry: separate production/destruction accumulation can leave floating-point cancellation residuals.

These are mechanism and state calculations. The complete 90 ns pulse trajectory and its repeated-run speed remain unvalidated. `PlasmaReactor` remains an isothermal model and does not accept Boltzmann phases requiring energy coupling.

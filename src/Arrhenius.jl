module Arrhenius
    using YAML
    using NPZ
    using SHA
    using LinearAlgebra
    using SparseArrays

    include("Constants.jl")
    include("BlowersMasel.jl")
    include("DataStructure.jl")
    include("Solution.jl")
    include("Magic.jl")
    include("Thermo.jl")
    include("Kinetics.jl")
    include("Transport.jl")
    include("MulticomponentTransport.jl")
    include("Equilibrium.jl")
    include("Reactors.jl")
    include("ReactorNetworks.jl")
    include("SurfaceKinetics.jl")
    include("PureWater.jl")
    include("RealGasThermo.jl")
    include("Flames.jl")
    include("FlameIO.jl")
    include("CounterflowFlames.jl")
    include("PremixedCounterflowFlames.jl")
    include("Precision.jl")
end

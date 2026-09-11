module Arrhenius
    using YAML
    using NPZ
    using SHA
    using LinearAlgebra
    using SparseArrays

    include("Constants.jl")
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
    include("PureWater.jl")
    include("RealGasThermo.jl")
    include("Flames.jl")
    include("FlameIO.jl")
    include("CounterflowFlames.jl")
    include("Precision.jl")
end

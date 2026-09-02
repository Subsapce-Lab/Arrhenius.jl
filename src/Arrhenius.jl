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
    include("Precision.jl")
end

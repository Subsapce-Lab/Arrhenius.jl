using Arrhenius, Test, SparseArrays

function reference_mass_action_product(C, species, orders, stoich, column, entries)
    standard = Arrhenius._mass_action_standard(orders, stoich, species, column, entries)
    nterms = 0
    if standard
        @inbounds for j in entries
            nterms += Int(stoich[species[j], column])
        end
    end
    standard &= 1 <= nterms <= 3
    multiplier = isempty(entries) ? one(eltype(C)) : one(C[species[first(entries)]])
    if standard
        negative_factors = 0
        @inbounds for j in entries
            c = C[species[j]]
            for _ in 1:Int(stoich[species[j], column])
                negative_factors += c < 0
                multiplier *= c
            end
        end
        return negative_factors >= 2 ? zero(multiplier) : multiplier
    end
    @inbounds for j in entries
        order = orders[j]
        order == 0 && continue
        c = C[species[j]]
        c > 0 || return zero(multiplier)
        multiplier *= c^order
    end
    return multiplier
end

@testset "mass action standard-order scan preserves baseline values" begin
    species = [1, 2]
    cases = (
        (Float64[0.3, 0.7], Float64[1, 1], sparse([1,2],[1,1],[1.,1.],2,1), 0.21),
        (Float64[-0.3, 0.7], Float64[1, 1], sparse([1,2],[1,1],[1.,1.],2,1), -0.21),
        (Float64[-0.3, -0.7], Float64[1, 1], sparse([1,2],[1,1],[1.,1.],2,1), 0.0),
        (Float64[0.3, 0.7], Float64[0.5, 1], sparse([1,2],[1,1],[1.,1.],2,1), 0.7sqrt(0.3)),
    )
    for (C, orders, stoich, expected) in cases
        entries = 1:length(species)
        actual = Arrhenius._mass_action_product(C, species, orders, stoich, 1, entries)
        baseline = reference_mass_action_product(C, species, orders, stoich, 1, entries)
        @test actual == baseline
        @test actual ≈ expected rtol=1e-15 atol=0
    end
end

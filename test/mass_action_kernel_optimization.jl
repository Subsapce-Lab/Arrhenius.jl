using Arrhenius, Test, SparseArrays

# Literal reference copy of the PR-base implementation at ddc0ab4.
@inline function old_mass_action_standard(orders, stoich, species, column, entries)
    length(entries) <= 3 || return false
    @inbounds for j in entries
        coefficient = stoich[species[j], column]
        (orders[j] == coefficient && coefficient >= 0 && isinteger(coefficient)) || return false
    end
    return true
end

@inline function old_mass_action_product(C, species, orders, stoich, column, entries)
    standard = old_mass_action_standard(orders, stoich, species, column, entries)
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

function comparison_vectors(ns)
    values = (0.0, 1.0e-25, -1.0e-25, -1.0e-14, 0.25, 1.0)
    mixed = [values[mod1(i, length(values))] for i in 1:ns]
    ordinary = [0.125 + 0.03125 * mod(i, 17) for i in 1:ns]
    return (zeros(ns), fill(1.0e-25, ns), fill(-1.0e-25, ns),
        fill(-1.0e-14, ns), mixed, ordinary)
end

function check_mass_action_mechanism(path)
    solution = Arrhenius.CreateSolution(path)
    reaction = solution.reaction
    reactant_species = rowvals(reaction.reactant_orders)
    reactant_orders = nonzeros(reaction.reactant_orders)
    product_species = rowvals(reaction.product_stoich_coeffs)
    product_orders = nonzeros(reaction.product_stoich_coeffs)
    reactant_stoich_rows = rowvals(reaction.reactant_stoich_coeffs)
    reactant_stoich_entries = reaction.reactant_stoich_coeffs
    reactant_entries = reaction.reactant_orders
    product_entries = reaction.product_stoich_coeffs
    vectors = comparison_vectors(solution.n_species)
    forward_count = 0
    reverse_count = 0

    # Sparse CSC matrices have unique row indices and no stored numeric zeros
    # after sparse(dense) construction. Assert this for every mechanism.
    @test all(x -> !iszero(x), nonzeros(reactant_entries))
    @test all(x -> !iszero(x), nonzeros(reactant_stoich_entries))
    @test all(x -> !iszero(x), nonzeros(product_entries))
    for col in 1:reaction.n_reactions
        rr = nzrange(reactant_entries, col)
        pr = nzrange(product_entries, col)
        sr = nzrange(reactant_stoich_entries, col)
        @test issorted(view(reactant_stoich_rows, sr))
        @test length(unique(view(reactant_stoich_rows, sr))) == length(sr)
        @test issorted(view(reactant_species, rr))
        @test length(unique(view(reactant_species, rr))) == length(rr)
        @test issorted(view(product_species, pr))
        @test length(unique(view(product_species, pr))) == length(pr)
        for (state_index, C) in enumerate(vectors)
            old = old_mass_action_product(C, reactant_species, reactant_orders,
                reaction.reactant_stoich_coeffs, col, rr)
            new = Arrhenius._mass_action_product(C, reactant_species, reactant_orders,
                reaction.reactant_stoich_coeffs, col, rr)
            old === new || error("forward mismatch: $path reaction=$col state=$state_index old=$old new=$new")
            forward_count += 1

            if reaction.is_reversible[col]
                old_reverse = old_mass_action_product(C, product_species, product_orders,
                    product_entries, col, pr)
                new_reverse = Arrhenius._mass_action_product(C, product_species, product_orders,
                    product_entries, col, pr)
                old_reverse === new_reverse || error("reverse mismatch: $path reaction=$col state=$state_index old=$old_reverse new=$new_reverse")
                reverse_count += 1
            end
        end
    end
    println("MASS_ACTION_MECHANISM path=$path species=$(solution.n_species) reactions=$(reaction.n_reactions) states=$(length(vectors)) forward=$forward_count reversible_reverse=$reverse_count")
    return (forward_count, reverse_count)
end

@testset "fused mass-action scan is bit-identical to PR-base helper" begin
    configured = get(ENV, "ARRHENIUS_MASS_ACTION_MECHANISMS", "")
    if isempty(configured)
        @testset "focused signed and fractional orders" begin
            species = [1, 2]
            cases = (
                (Float64[0.3, 0.7], Float64[1, 1], sparse([1,2],[1,1],[1.,1.],2,1)),
                (Float64[-0.3, 0.7], Float64[1, 1], sparse([1,2],[1,1],[1.,1.],2,1)),
                (Float64[-0.3, -0.7], Float64[1, 1], sparse([1,2],[1,1],[1.,1.],2,1)),
                (Float64[0.3, 0.7], Float64[0.5, 1], sparse([1,2],[1,1],[1.,1.],2,1)),
            )
            for (C, orders, stoich) in cases
                entries = 1:length(species)
                @test Arrhenius._mass_action_product(C, species, orders, stoich, 1, entries) ===
                    old_mass_action_product(C, species, orders, stoich, 1, entries)
            end
        end
    else
        paths = split(configured, Sys.iswindows() ? ';' : ':')
        @test all(isfile, paths)
        forward_total = 0
        reverse_total = 0
        for path in paths
            forward, reverse = check_mass_action_mechanism(path)
            forward_total += forward
            reverse_total += reverse
        end
        println("MASS_ACTION_TOTAL mechanisms=$(length(paths)) forward=$forward_total reversible_reverse=$reverse_total")
    end
end

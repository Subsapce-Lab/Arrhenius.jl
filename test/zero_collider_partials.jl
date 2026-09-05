using SparseArrays

@testset "exact partials at zero collider" begin
    for family in (:third_body, :lindemann, :troe), reversible in (false, true)
        reactants = sparse(reshape([1.0, 0.0], 2, 1))
        products = sparse(reshape([0.0, 1.0], 2, 1))
        reaction = Arrhenius.Reaction(
            products, reactants, copy(reactants), [reversible],
            reshape([2.0, 0.0, 0.0], 1, 3),
            reshape([3.0, 0.0, 0.0], 1, 3),
            reshape([0.5, 1000.0, 2000.0, 3000.0], 1, 4),
            family == :third_body ? [1] : Int[],
            family == :third_body ? Int[] : [1],
            family == :third_body ? Int[] : [family == :troe ? 1 : -1],
            sparse(reshape([0.0, 1.0], 2, 1)), [[1]], [[2]], 1,
            products - reactants, [0.0],
            Arrhenius.PlogData(Int[], Int[], Int[], Float64[], Int[], zeros(0, 3)),
        )
        workspace = KineticsDerivativeWorkspace(reaction)
        qdot, dC, dT = zeros(1), zeros(1, 2), zeros(1)
        for multiplier in (1.0, 2.5), collider in (0.0, 1e-6, 1.0)
            concentrations = [2.0, collider]
            function rate(state)
                return wdot_func(reaction, state[end], state[1:2], zeros(2),
                    zeros(2); get_qdot=true, rate_multipliers=[multiplier])
            end
            state = vcat(concentrations, 1000.0)
            reference = ForwardDiff.jacobian(rate, state)
            reaction_rate_partials!(qdot, dC, dT, reaction, 1000.0,
                concentrations, zeros(2), zeros(2), workspace;
                rate_multipliers=[multiplier])
            @test qdot ≈ rate(state) rtol=1e-12 atol=1e-14
            @test hcat(dC, dT) ≈ reference rtol=1e-10 atol=1e-12
        end
    end
end

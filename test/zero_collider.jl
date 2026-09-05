using Arrhenius
using ForwardDiff
using Test

@testset "falloff AD at zero collider" begin
    for troe in (false, true)
        reaction = (
            n_reactions=1, Arrhenius_coeffs=reshape([2.0, 0.0, 0.0], 1, 3),
            Arrhenius_0=reshape([3.0, 0.0, 0.0], 1, 3),
            plog=(reaction_indices=Int[],), index_three_body=Int[],
            index_falloff=[1], index_falloff_Troe=[troe ? 1 : -1],
            Troe_=reshape([0.5, 1000.0, 2000.0, 3000.0], 1, 4),
            efficiencies_coeffs=reshape([0.0, 1.0], 2, 1),
            vk=reshape([-1.0, 1.0], 2, 1), vk_sum=[0.0],
            is_reversible=[false], i_reactant=[[1]], i_product=[[2]],
            reactant_orders=reshape([1.0, 0.0], 2, 1),
            product_stoich_coeffs=reshape([0.0, 1.0], 2, 1),
        )
        function rate(C)
            work = KineticsWorkspace(ntuple(_ -> zeros(eltype(C), 1), 6)...)
            return wdot!(similar(C), reaction, 1000.0, C, zeros(2), zeros(2),
                work; get_qdot=true)
        end
        Fcent = 0.5 * exp(-1 / 3) + 0.5 * exp(-1.0) + exp(-2.0)
        slope = 6.0 * (troe ? Fcent^(1 / (1 + (1 / 0.14)^2)) : 1.0)
        @test only(rate([2.0, 0.0])) == 0.0
        jac = ForwardDiff.jacobian(rate, [2.0, 0.0])
        @test all(isfinite, jac)
        @test jac[1, 1] == 0.0
        @test jac[1, 2] ≈ slope rtol=1e-14
        @test all(isfinite, ForwardDiff.jacobian(rate, [0.0, 0.0]))
        @test all(iszero, ForwardDiff.jacobian(rate, [0.0, 0.0]))
        @test ForwardDiff.derivative(x -> only(rate([2.0, x])), -1e-6) == 0.0
        # Positive-concentration interior agrees with a centered finite difference.
        for collider in (1e-6, 1.0, 1e6)
            step = collider * 1e-4
            fd = (only(rate([2.0, collider + step])) - only(rate([2.0, collider - step]))) / (2step)
            ad = ForwardDiff.derivative(x -> only(rate([2.0, x])), collider)
            @test ad ≈ fd rtol=5e-5 atol=1e-12
        end
    end
end

@testset "low-order concentration powers" begin
    power = Arrhenius._concentration_power
    for Scalar in (Float32, Float64)
        for x in Scalar.((-2, -0.0, 0.0, 1.0e-15, 0.1, 1, 2, 1.0e10))
            for order in Scalar.((0, 1, 2, 3, 4))
                @test isequal(power(x, order), x^order)
                @test typeof(power(x, order)) === typeof(x^order)
            end
        end
        for x in Scalar.((0.1, 1, 4)), order in Scalar.((-1, 0.5, 1.5, 2.5))
            @test power(x, order) == x^order
        end
        @test_throws DomainError power(Scalar(-1), Scalar(1.5))
        for x in Scalar.((Inf, -Inf, NaN)), order in Scalar.((1, 2))
            @test isequal(power(x, order), x^order)
        end
        for x in Scalar.((-2, 0, 0.1, 2)), order in Scalar.((1, 2))
            expected = order == one(order) ? one(x) : 2x
            @test ForwardDiff.derivative(c -> power(c, order), x) ≈ expected
            @test ForwardDiff.derivative(c -> ForwardDiff.derivative(
                d -> power(d, order), c), x) ≈ (order == one(order) ? zero(x) : Scalar(2))
        end
    end
    # Promotion and rate-order derivatives must not disappear at integer values.
    for x in (Float32(0.2), 0.2), order in (Float32(1), 1.0, Float32(2), 2.0)
        @test power(x, order) ≈ x^order
        @test typeof(power(x, order)) === typeof(x^order)
    end
    for order in (1.0, 2.0, 1.5)
        @test ForwardDiff.derivative(p -> power(2.0, p), order) ≈ 2.0^order * log(2.0)
    end
end

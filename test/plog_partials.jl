using SparseArrays

@testset "PLOG temperature and pressure derivatives" begin
    plog = Arrhenius.PlogData([1], [0], [1, 4], [1e5, 1e6, 1e7],
        [1, 3, 4, 5], [2.0 0.2 50.0; 3.0 -0.1 100.0; 20.0 0.3 20.0; 40.0 0.1 10.0])
    for temperature in (700.0, 1200.0), pressure in (5e4, 3e5, 3e6, 2e7)
        rate, dT, dP = Arrhenius._plog_rate_and_derivatives(plog, 1, temperature, pressure)
        f(state) = Arrhenius._plog_rate(plog, 1, state[1], state[2], log(state[1]))
        reference = ForwardDiff.gradient(f, [temperature, pressure])
        @test rate ≈ f([temperature, pressure]) rtol=1e-12
        @test [dT, dP] ≈ reference rtol=1e-11 atol=1e-12
    end
    # Knots are piecewise differentiable: endpoints select the constant-pressure
    # exterior branch; the interior knot uses the lower interval, matching RHS.
    for pressure in plog.pressures
        rate, _, dP = Arrhenius._plog_rate_and_derivatives(plog, 1, 1000.0, pressure)
        @test rate ≈ Arrhenius._plog_rate(plog, 1, 1000.0, pressure, log(1000.0))
        if pressure == first(plog.pressures) || pressure == last(plog.pressures)
            @test dP == 0.0
        else
            h = pressure * 1e-6
            fd = (rate - Arrhenius._plog_rate(plog, 1, 1000.0, pressure-h, log(1000.0))) / h
            @test dP ≈ fd rtol=1e-5
        end
    end

    reactants = sparse(reshape([1.0, 0.0], 2, 1))
    products = sparse(reshape([0.0, 1.0], 2, 1))
    for collider in (0, 2), reversible in (false, true), multiplier in (1.0, 2.5)
        local_plog = Arrhenius.PlogData([1], [collider], plog.group_offsets,
            plog.pressures, plog.rate_offsets, plog.Arrhenius_coeffs)
        reaction = Arrhenius.Reaction(products, reactants, copy(reactants), [reversible],
            reshape([1.0, 0.0, 0.0], 1, 3), zeros(0,3), zeros(0,4), Int[], Int[], Int[],
            spzeros(2,1), [[1]], [[2]], 1, products-reactants, [0.0], local_plog)
        work = KineticsDerivativeWorkspace(reaction)
        qdot, dC, dT = zeros(1), zeros(1,2), zeros(1)
        for concentration in ([0.02, 0.0], [0.02, 0.015])
            state = vcat(concentration, 1000.0)
            f(u) = wdot_func(reaction, u[end], u[1:2], zeros(2), zeros(2);
                get_qdot=true, rate_multipliers=[multiplier])
            reaction_rate_partials!(qdot, dC, dT, reaction, 1000.0, concentration,
                zeros(2), zeros(2), work; rate_multipliers=[multiplier])
            @test qdot ≈ f(state) rtol=1e-12 atol=1e-14
            @test hcat(dC,dT) ≈ ForwardDiff.jacobian(f,state) rtol=1e-11 atol=1e-12
        end
    end
end

@testset "constant-pressure exact reactor Jacobian" begin
    gas = CreateSolution(joinpath(@__DIR__, "..", "mechanism", "gri30.yaml"))
    ns = gas.n_species
    work = ConstantPressureJacobianWorkspace(gas)
    J, rhs = zeros(ns+1,ns+1), zeros(ns+1)
    function reference(state, pressure, multipliers)
        Y, temperature = state[1:ns], state[end]
        mean_mw = inv(sum(Y ./ gas.MW))
        density = pressure * mean_mw / (R * temperature)
        X = Y .* mean_mw ./ gas.MW
        C = density .* Y ./ gas.MW
        cp = cal_cp_R(gas, temperature, pressure, X)
        h = R * temperature .* cal_h_RT(gas, temperature, pressure, X)
        s = R .* cal_s0_R(gas, temperature, pressure, X)
        source = wdot_func(gas.reaction, temperature, C, s, h; rate_multipliers=multipliers)
        cp_mass = R * sum(cp .* Y ./ gas.MW)
        return vcat(gas.MW .* source ./ density, -dot(h,source)/(density*cp_mass))
    end
    for temperature in (700.0, 1200.0, 2400.0), pressure in (1e5, 2e6)
        # Deliberately do not normalize: all species are independent state inputs.
        Y = collect(1.0:ns) ./ ns^2
        state = vcat(Y,temperature)
        multipliers = 0.5 .+ collect(1.0:gas.n_reactions) ./ gas.n_reactions
        constant_pressure_jacobian!(J,rhs,gas,pressure,state,work; rate_multipliers=multipliers)
        expected = reference(state,pressure,multipliers)
        expected_J = ForwardDiff.jacobian(u -> reference(u,pressure,multipliers),state)
        @test norm(rhs-expected,Inf)/norm(expected,Inf) < 1e-11
        @test norm(J-expected_J,Inf)/norm(expected_J,Inf) < 1e-10
        # A mixed species/temperature direction also checks density coupling.
        direction = vcat(sin.(collect(1:ns)) .* Y, temperature/10)
        step = 1e-5
        fd = (reference(state+step*direction,pressure,multipliers) -
            reference(state-step*direction,pressure,multipliers))/(2step)
        @test norm(J*direction-fd,Inf)/norm(fd,Inf) < 1e-7
        # Accepted ODE states and Newton iterates may contain tiny negatives.
        # Match the existing RHS extension without silently projecting the state.
        signed_state = copy(state)
        signed_state[1] = -1e-14
        @test_throws DomainError constant_pressure_jacobian!(J,rhs,gas,pressure,signed_state,work)
        constant_pressure_jacobian!(J,rhs,gas,pressure,signed_state,work;
            rate_multipliers=multipliers, allow_signed_state=true)
        signed_rhs = reference(signed_state,pressure,multipliers)
        signed_J = ForwardDiff.jacobian(u -> reference(u,pressure,multipliers),signed_state)
        @test norm(rhs-signed_rhs,Inf)/norm(signed_rhs,Inf) < 1e-11
        @test norm(J-signed_J,Inf)/norm(signed_J,Inf) < 1e-10
    end
    @test_throws DomainError constant_pressure_jacobian!(J,rhs,gas,1e5,zeros(ns+1),work)
end

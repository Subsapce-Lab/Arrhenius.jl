using Arrhenius, Test

@testset "Newton policy option resolution" begin
    gas=CreateSolution(joinpath(@__DIR__,"..","mechanism","h2o2.yaml"))
    X=mole_fractions(gas,"H2:2,O2:1,N2:3")
    f=FreeFlame(gas;X,grid=collect(range(0.,1.;length=5)),discretization=:finite_difference)
    current=Arrhenius._newton_policy(nothing,false)
    compatible=Arrhenius._newton_policy(nothing,true)

    @test current==NewtonPolicyOptions()
    @test current.cantera_compatible == false
    @test current.damping_trials==7 && current.damping_factor==sqrt(2.)
    @test current.steady_convergence===:residual_inf
    @test (current.premixed_steady_maxiters,current.premixed_policy_transient_maxiters)==(35,200)
    @test current.jacobian_age==20 && current.jacobian_reevaluations==4

    @test compatible.cantera_compatible
    @test compatible.damping_acceptance===:step_or_reduction
    @test compatible.step_bounds===:cantera
    @test compatible.norm_denominator===:cantera && compatible.flow_weight===:velocity
    @test compatible.steady_convergence===:weighted_step
    @test all(iszero,(compatible.premixed_steady_maxiters,
        compatible.counterflow_steady_maxiters,compatible.premixed_transient_maxiters,
        compatible.premixed_policy_transient_maxiters,compatible.counterflow_transient_maxiters))

    @test !Arrhenius._newton_merit_accepts(current,0.8,0.9,false)
    @test Arrhenius._newton_merit_accepts(compatible,0.8,0.9,false)
    reduction=NewtonPolicyOptions(damping_acceptance=:correction_reduction)
    @test !Arrhenius._newton_merit_accepts(reduction,0.8,0.9,true)
    custom=Arrhenius._newton_policy(NewtonPolicyOptions(damping_trials=3,damping_factor=2.0,
        jacobian_age=9,jacobian_reevaluations=2),false)
    @test (custom.damping_trials,custom.damping_factor,custom.jacobian_age,
        custom.jacobian_reevaluations)==(3,2.0,9,2)
    @test Arrhenius._cantera_flame_temperature_bounds(f) != (.2,6.0)

    # On the five-point h2o2 state, the flow-weight and norm-divisor switches
    # select measurably different correction metrics.
    wcurrent=Arrhenius._flame_correction_weights(f,f.state,false,current)
    wvelocity=Arrhenius._flame_correction_weights(f,f.state,false,compatible)
    @test wcurrent[end] != wvelocity[end]
    step=ones(size(f.state))
    ncurrent=Arrhenius._flame_policy_correction_norm(step,wcurrent,f,current)
    ncompatible=Arrhenius._flame_policy_correction_norm(step,wvelocity,f,compatible)
    @test ncurrent != ncompatible
    @test_throws ArgumentError solve!(f;newton_options=:invalid)
end

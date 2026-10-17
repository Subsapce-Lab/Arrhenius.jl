using Arrhenius, Test

@testset "Newton policy option resolution" begin
    gas=CreateSolution(joinpath(@__DIR__,"..","mechanism","h2o2.yaml"))
    X=mole_fractions(gas,"H2:2,O2:1,AR:3")
    f=FreeFlame(gas;X,grid=collect(range(0.,1.;length=5)),discretization=:finite_difference)
    current=Arrhenius._newton_policy(nothing,false)
    compatible=Arrhenius._newton_policy(nothing,true)

    @test current.cantera_compatible == false
    @test (current.damping_trials,current.damping_factor,current.damping_acceptance)==(7,sqrt(2.),:current)
    @test (current.step_bounds,current.norm_denominator,current.flow_weight,current.steady_convergence)==
        (:current,:current,:current,:residual_inf)
    @test (current.premixed_steady_maxiters,current.counterflow_steady_maxiters,
        current.premixed_counterflow_steady_maxiters,current.premixed_transient_maxiters,
        current.premixed_policy_transient_maxiters,current.counterflow_transient_maxiters)==(35,45,0,20,200,100)
    @test current.jacobian_age==20 && current.jacobian_reevaluations==4

    expected_cantera=(damping_trials=7,damping_factor=sqrt(2.),
        damping_acceptance=:step_or_reduction,step_bounds=:cantera,
        norm_denominator=:cantera,flow_weight=:velocity,
        steady_convergence=:weighted_step,premixed_steady_maxiters=0,
        counterflow_steady_maxiters=0,premixed_counterflow_steady_maxiters=0,
        premixed_transient_maxiters=0,premixed_policy_transient_maxiters=0,
        counterflow_transient_maxiters=0,jacobian_age=20,jacobian_reevaluations=4)
    @test compatible.cantera_compatible
    @test all(getproperty(compatible,k)==v for (k,v) in pairs(expected_cantera))

    # Every field supplied by the caller overlays the compatible preset alone.
    overrides=(damping_trials=3,damping_factor=2.0,
        damping_acceptance=:correction_reduction,step_bounds=:current,
        norm_denominator=:current,flow_weight=:current,
        steady_convergence=:residual_inf,premixed_steady_maxiters=1,
        counterflow_steady_maxiters=2,premixed_counterflow_steady_maxiters=3,
        premixed_transient_maxiters=4,premixed_policy_transient_maxiters=5,
        counterflow_transient_maxiters=6,jacobian_age=9,jacobian_reevaluations=2)
    for (field,value) in pairs(overrides)
        option=NewtonPolicyOptions(cantera_compatible=true; (field=>value,)...)
        resolved=Arrhenius._newton_policy(option,false)
        @test resolved.cantera_compatible
        @test getproperty(resolved,field)==value
        for (other,preset_value) in pairs(expected_cantera)
            other==field || @test getproperty(resolved,other)==preset_value
        end
    end
    @test Arrhenius._newton_policy(NewtonPolicyOptions(cantera_compatible=true,damping_trials=3),false).damping_trials==3
    @test Arrhenius._newton_policy(NewtonPolicyOptions(damping_trials=3),true).damping_trials==3

    @test !Arrhenius._newton_merit_accepts(current,0.95,0.9,false)
    @test Arrhenius._newton_merit_accepts(compatible,0.95,0.9,false)
    reduction=NewtonPolicyOptions(damping_acceptance=:correction_reduction)
    @test !Arrhenius._newton_merit_accepts(Arrhenius._newton_policy(reduction,false),0.95,0.9,true)
    @test Arrhenius._cantera_damped_step_converged(true,1.0)
    @test Arrhenius._cantera_damped_step_converged(true,0.9)
    @test !Arrhenius._cantera_damped_step_converged(false,0.9)
    @test !Arrhenius._cantera_damped_step_converged(true,1.000001)
    # At s1 == 1, Cantera accepts only when the correction also decreases;
    # convergence then follows from the accepted damping outcome.
    @test Arrhenius._newton_merit_accepts(compatible,1.0,1.1,true)
    @test !Arrhenius._newton_merit_accepts(compatible,1.0,0.9,true)

    # Record actual one-iteration traces on five-point premixed and counterflow
    # flames. The independent hand trace applies MultiNewton's accepted && s1<=1 rule.
    premixed_trace=Tuple{Bool,Float64,Float64,Bool}[]
    Arrhenius._flame_newton!(f,Arrhenius.FlameWorkspace(f);maxiters=1,
        newton_options=compatible,step_trace=premixed_trace)
    @test !isempty(premixed_trace)
    @test [(isfinite(s1)&&(s1<1 || s1<s0),isfinite(s1)&&(s1<1 || s1<s0)&&s1<=1)
        for (_,s0,s1,_) in premixed_trace]==[(a,c) for (a,_,_,c) in premixed_trace]

    cf=CounterflowDiffusionFlame(gas;fuel="H2:1",oxidizer="O2:1,AR:3",
        mdot_fuel=.12,mdot_oxidizer=.12,P=.05*one_atm,width=.02,
        grid=collect(range(0.,.02;length=5)))
    counterflow_trace=Tuple{Bool,Float64,Float64,Bool}[]
    Arrhenius._counterflow_newton!(cf,Arrhenius.CounterflowWorkspace(cf);maxiters=1,
        analytic=false,newton_options=compatible,step_trace=counterflow_trace)
    @test !isempty(counterflow_trace)
    @test [(isfinite(s1)&&(s1<1 || s1<s0),isfinite(s1)&&(s1<1 || s1<s0)&&s1<=1)
        for (_,s0,s1,_) in counterflow_trace]==[(a,c) for (a,_,_,c) in counterflow_trace]

    wcurrent=Arrhenius._flame_correction_weights(f,f.state,false,current)
    wvelocity=Arrhenius._flame_correction_weights(f,f.state,false,compatible)
    @test wcurrent[end] != wvelocity[end]
    step=ones(size(f.state))
    ncurrent=Arrhenius._flame_policy_correction_norm(step,wcurrent,f,current)
    ncompatible=Arrhenius._flame_policy_correction_norm(step,wvelocity,f,compatible)
    @test ncurrent != ncompatible
    @test_throws ArgumentError solve!(f;newton_options=:invalid)
end

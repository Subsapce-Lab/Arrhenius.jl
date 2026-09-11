using SciMLBase, OrdinaryDiffEqSDIRK

"Native L-stable SDIRK adapter for the stiff shock-tube examples."
function shocktube_sdirk(problem;kwargs...)
    f = ODEFunction(problem.f;jac=problem.jac,tgrad=problem.tgrad)
    ode = ODEProblem(f,problem.u0,problem.tspan,problem.p)
    outside_domain = (u,p,t) -> !all(isfinite,u) || u[end] <= 0 ||
        minimum(view(u,1:length(u)-1)) < -1e-13
    solution = solve(ode,KenCarp4();isoutofdomain=outside_domain,kwargs...)
    SciMLBase.successful_retcode(solution) || error("shock-tube integration failed: $(solution.retcode)")
    return solution
end

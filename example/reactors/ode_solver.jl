using SciMLBase
using OrdinaryDiffEqBDF

"Integrate an Arrhenius reactor problem with Julia's adaptive QNDF solver."
function native_bdf(problem; kwargs...)
    f = ODEFunction(problem.f; jac=problem.jac, tgrad=problem.tgrad)
    ode = ODEProblem(f, problem.u0, problem.tspan, problem.p)
    outside_domain = (u, p, t) -> !all(isfinite, u) || u[end] <= 0 ||
        minimum(view(u, 1:length(u)-1)) < -1e-13
    solution = solve(ode, QNDF(); isoutofdomain=outside_domain, kwargs...)
    SciMLBase.successful_retcode(solution) || error("reactor integration failed: $(solution.retcode)")
    return solution
end

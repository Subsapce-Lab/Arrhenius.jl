module ArrheniusMetalExt

using Arrhenius
using Metal

backend_status() = :solver_not_connected

function prepare_ensemble(
    backend,
    mechanism,
    reactor,
    policies,
    sample_solver,
)
    throw(Arrhenius.BackendUnavailableError(
        :metal,
        "ArrheniusMetalExt is loaded, but a Metal ensemble solver has not yet " *
        "been connected",
    ))
end

end

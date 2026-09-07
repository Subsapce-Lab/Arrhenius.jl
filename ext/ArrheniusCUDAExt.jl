module ArrheniusCUDAExt

using Arrhenius
using CUDA

backend_status() = :solver_not_connected

function prepare_ensemble(
    backend,
    mechanism,
    reactor,
    policies,
    sample_solver,
)
    throw(Arrhenius.BackendUnavailableError(
        :cuda,
        "ArrheniusCUDAExt is loaded, but a CUDA ensemble solver has not yet " *
        "been connected",
    ))
end

end

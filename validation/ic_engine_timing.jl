# Full shared ic_engine calculation, including fresh mechanism/network creation,
# accepted-state quadrature and output quantities. Imports, artifact I/O and
# repetition checks are outside timers. No solver equations are copied here.
# julia --project=SOLVER_ENV validation/ic_engine_timing.jl MECHANISM.yaml OUTPUT.npz
#       [REPETITIONS=9] [informational|controlled|validate-only]
# SOLVER_ENV requires SciMLBase, OrdinaryDiffEqSDIRK and NPZ.
import_seconds = @elapsed begin
    @eval using Arrhenius, NPZ, SHA, Dates, TOML, LinearAlgebra, Libdl
    include(joinpath(@__DIR__,"..","example","reactors","ic_engine_solver.jl"))
    include(joinpath(@__DIR__,"..","example","reactors","ic_engine_setup.jl"))
    include("numerical_threads.jl")
end

function engine_timing_calculation(mechanism)
    result = solve_ic_engine(mechanism;integrator=native_engine_sdirk)
    output = ic_engine_observables(result)
    return (;result,output,summary=ic_engine_summary(result,output))
end

function engine_timing_checks(calculation)
    s = calculation.summary
    return all(isfinite,reduce(vcat,calculation.result.states)) &&
        s["end_time_s"] == 0.16 && s["integration_points"] > s["display_points"] > 2880 &&
        s["maximum_output_interval_s"] <= (1/(360*50))*(1+1e-12) &&
        s["maximum_output_temperature_change_K"] <= 20+1e-8 &&
        s["volume_identity_error_m3"] < 1e-10 && s["mass_balance_relative_drift"] < 2e-7 &&
        s["energy_balance_relative_drift"] < 2e-6 && s["minimum_mass_fraction"] >= -1e-13 &&
        maximum(values(s["relative_quadrature_change"])) < 1e-4 &&
        s["work_quadrature_ledger_relative_error"] < 1e-4
end

function engine_require_checks(calculation,label)
    engine_timing_checks(calculation) || error("$label failed engine physical/integral checks")
    return true
end
function engine_require_replay(actual,expected,label)
    isequal(actual.result.states,expected.result.states) &&
        isequal(actual.output,expected.output) && isequal(actual.summary,expected.summary) ||
        error("$label differs from the checked first engine trajectory")
    return true
end

function engine_source_hashes(root)
    paths = [joinpath(folder,file) for (folder,_,files) in walkdir(joinpath(root,"src"))
             for file in files if endswith(file,".jl")]
    append!(paths,[joinpath(root,"example","reactors",file)
        for file in ("ic_engine_setup.jl","ic_engine_solver.jl","ic_engine.jl")])
    push!(paths,joinpath(root,"validation","numerical_threads.jl"))
    return Dict(relpath(path,root)=>bytes2hex(sha256(read(path))) for path in paths)
end

function engine_timing_main(args)
    length(args)>=2 || error("supply MECHANISM.yaml OUTPUT.npz [REPETITIONS] [QUALIFICATION]")
    mechanism,output_path = args[1:2]
    repetitions = length(args)>=3 ? parse(Int,args[3]) : 9
    qualification = length(args)>=4 ? args[4] : "informational"
    repetitions>=9 || error("at least nine warm repetitions required")
    qualification in ("informational","controlled","validate-only") || error("invalid qualification")
    root = normpath(joinpath(@__DIR__,".."))
    realpath(pathof(Arrhenius))==realpath(joinpath(root,"src","Arrhenius.jl")) ||
        error("the benchmark and Arrhenius package must use the same checkout")
    source_hashes = engine_source_hashes(root)
    thread_keys = Sys.isapple() ? ("JULIA_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS") :
                                 ("JULIA_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")
    thread_environment = Dict(k=>get(ENV,k,"") for k in thread_keys)
    if qualification == "controlled"
        Threads.nthreads()==1 && BLAS.get_num_threads()==1 && all(==("1"),values(thread_environment)) ||
            error("controlled timing requires one Julia/BLAS thread and all recorded thread variables set to 1")
    end
    thread_checks = Dict("before_first"=>benchmark_julia_thread_settings())
    started = time_ns()
    first = engine_timing_calculation(mechanism)
    first_seconds = (time_ns()-started)/1e9
    first_checked = engine_require_checks(first,"first invocation")
    thread_checks["before_warm"] = benchmark_julia_thread_settings()
    println("Engine first calculation: ",first_seconds," s including JIT; checks ",first_checked); flush(stdout)
    samples,matches,checks = Float64[],Bool[],Bool[]
    if qualification != "validate-only"
        GC.gc()
        for repetition in 1:repetitions
            started = time_ns()
            repeated = engine_timing_calculation(mechanism)
            push!(samples,(time_ns()-started)/1e9)
            thread_checks["after_warm_"*string(repetition)] = benchmark_julia_thread_settings(;enforce=false)
            # Full physical states, every output quantity and all integral /
            # convergence diagnostics must replay the checked first solution.
            push!(matches,engine_require_replay(repeated,first,"warm repetition $repetition"))
            push!(checks,engine_require_checks(repeated,"warm repetition $repetition"))
            println("Engine warm repetition ",repetition,": ",last(samples)," s; checks ",last(matches)&&last(checks)); flush(stdout)
        end
    end
    thread_checks["after_all"] = benchmark_julia_thread_settings(;enforce=false)
    hashes_unchanged = source_hashes == engine_source_hashes(root)
    cpu = Sys.isapple() ? readchomp(`sysctl -n machdep.cpu.brand_string`) : Sys.islinux() ?
        strip(split(Base.first(filter(l->startswith(l,"model name"),readlines("/proc/cpuinfo"))),':';limit=2)[2]) : Sys.CPU_NAME
    utf8(v) = collect(codeunits(string(v)))
    toml(v) = sprint(io->TOML.print(io,v;sorted=true))
    filehash(p) = bytes2hex(sha256(read(p)))
    libraries = Dict(realpath(path)=>filehash(path) for path in Libdl.dllist() if isfile(path))
    project = Base.active_project()
    manifest = joinpath(dirname(project),"Manifest.toml")
    data = Dict{String,Any}(
        "scope_utf8"=>utf8("full_source_eight_revolutions"),"qualification_utf8"=>utf8(qualification),
        "cpu_utf8"=>utf8(cpu),"system_utf8"=>utf8(Sys.isapple() ? "Darwin" : Sys.islinux() ? "Linux" : string(Sys.KERNEL)),
        "kernel_release_utf8"=>utf8(Sys.isunix() ? readchomp(`uname -r`) : "unknown"),
        "platform_utf8"=>utf8(Sys.MACHINE),"julia_version_utf8"=>utf8(VERSION),
        "timestamp_utc_utf8"=>utf8(Dates.now(Dates.UTC)),
        "thread_environment_toml_utf8"=>utf8(toml(thread_environment)),
        "thread_checks_toml_utf8"=>utf8(toml(thread_checks)),
        "thread_helper_sha256_utf8"=>utf8(filehash(joinpath(@__DIR__,"numerical_threads.jl"))),
        "source_hashes_toml_utf8"=>utf8(toml(source_hashes)),"source_hashes_unchanged"=>[UInt8(hashes_unchanged)],
        "julia_library_hashes_toml_utf8"=>utf8(toml(libraries)),
        "project_toml_utf8"=>utf8(read(project,String)),"manifest_toml_utf8"=>utf8(read(manifest,String)),
        "project_sha256_utf8"=>utf8(filehash(project)),"manifest_sha256_utf8"=>utf8(filehash(manifest)),
        "julia_executable_sha256_utf8"=>utf8(filehash(joinpath(Sys.BINDIR,Sys.iswindows() ? "julia.exe" : "julia"))),
        "harness_sha256_utf8"=>utf8(filehash(@__FILE__)),
        "mechanism_sha256_utf8"=>utf8(filehash(mechanism)),"sidecar_sha256_utf8"=>utf8(filehash(mechanism*".npz")),
        "solver_utf8"=>utf8("OrdinaryDiffEqSDIRK.KenCarp4; shared native_engine_sdirk adapter"),
        "summary_toml_utf8"=>utf8(toml(first.summary)),
        "julia_threads"=>[Threads.nthreads()],"blas_threads"=>[BLAS.get_num_threads()],
        "import_seconds"=>[import_seconds],"first_seconds"=>[first_seconds],"warm_seconds"=>samples,
        "first_checked"=>[UInt8(first_checked)],"warm_matches_first"=>UInt8.(matches),"warm_checked"=>UInt8.(checks),
        "native_rtol"=>[1e-13],"native_species_atol_kg"=>[1e-26],"native_temperature_atol_K"=>[1e-10],
        "native_volume_atol_m3"=>[1e-20],"states"=>reduce(hcat,first.result.states),
    )
    for (key,value) in first.output
        data["output_"*key] = value
    end
    npzwrite(output_path,data)
    first_checked && all(matches) && all(checks) && hashes_unchanged || error("engine correctness or repetition checks failed")
end

if abspath(PROGRAM_FILE) == @__FILE__
    engine_timing_main(ARGS)
end

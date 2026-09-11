"""Paired full-source ic_engine.py timing with independent physical validation.

Run ic_engine_timing.jl first on the same otherwise idle host. This driver
executes the calculation AST from the exact pinned Cantera source: mechanism
and network construction, all eight revolutions, output quantities and source
integral estimates. Imports, plotting, printing and artifact I/O are excluded.
The native timer calls the shared solve_ic_engine and observable/summary APIs,
including their converged accepted-state quadrature. The source's sparse heat
integral is retained as a source estimate, never used as the physical reference.

python validation/ic_engine_timing.py --julia-result native.npz --output paired.json \
    --native-mechanism dodecane_IG.yaml --source-example /source/reactors/ic_engine.py \
    --cantera-build-record build-record.json --target apple-m4 --repetitions 9
Use --validate-only for one nonqualifying invocation. Controlled qualification
requires both sides to record controlled mode and all environment/build checks.
"""
from pathlib import Path
import argparse
import ast
import copy
import gc
import hashlib
import json
import os
import statistics
import time
import tomllib
from benchmark_environment import host_metadata, matches_target, cantera_library_hashes, verify_numerical_threads

_started = time.perf_counter()
import numpy as np
import cantera as ct
import cantera._cantera as compiled
IMPORT_SECONDS = time.perf_counter()-_started
SOURCE_COMMIT = "726522be4e2a13454d8415b7ef799d621f665cf3"
SOURCE_SHA256 = "43acf803aa5e4589bb1e718a480cbcc7db4558e49cdb97bb5b7ff67d928a68bb"
MIN_REPETITIONS = 9
REFINED_RTOL, REFINED_ATOL = 1e-13, 1e-24
TRAPEZOID = getattr(np,"trapezoid",None) or np.trapz


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_source_checks(result, label):
    if not source_checks(result):
        raise RuntimeError(f"{label} failed source output checks")
    return True


def require_source_replay(actual, expected, label):
    if not output_replay(actual, expected):
        raise RuntimeError(f"{label} differs from the checked first source trajectory")
    return True


def require_native_replay_flags(native):
    lengths={len(native[key]) for key in ("warm_seconds","warm_matches_first","warm_checked")}
    if (len(lengths)!=1 or not native["first_checked"][0] or not native["source_hashes_unchanged"][0]
            or not all(native["warm_matches_first"]) or not all(native["warm_checked"])):
        raise RuntimeError("native artifact contains failed or incomplete first/warm checks")
    return True


def decode_metadata(archive):
    return {key[:-5]:bytes(archive[key]).decode() for key in archive.files if key.endswith("_utf8")}


def source_programs(path,mechanism):
    """Select original calculation statements without rewriting its physics."""
    if sha(path) != SOURCE_SHA256:
        raise ValueError("ic_engine.py does not match the pinned source SHA256")
    tree = ast.parse(Path(path).read_text())
    setup,loop = [],None
    for original in tree.body:
        node = copy.deepcopy(original)
        if isinstance(node,(ast.Import,ast.ImportFrom)) or (isinstance(node,ast.Expr) and
                isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)):
            continue
        if isinstance(node,ast.FunctionDef) and node.name=="ca_ticks":
            break
        if isinstance(node,ast.While):
            loop = node
            continue
        if loop is None:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="reaction_mechanism" for t in node.targets):
                node.value = ast.Constant(str(mechanism))
            setup.append(node)
    # Integral assignments occur after plots, so find their section separately.
    integrals=[]
    in_integrals=False
    for original in tree.body:
        if isinstance(original,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="Q" for t in original.targets):
            in_integrals=True
        if in_integrals and isinstance(original,(ast.Assign,ast.AugAssign)):
            integrals.append(copy.deepcopy(original))
    if loop is None or not integrals:
        raise ValueError("pinned source calculation/integral sections were not found")
    def code(nodes):
        return compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),"exec")
    return code(setup),code([loop]),code(ast.parse("t = states.t").body+integrals)


def new_source_state(programs):
    namespace={"ct":ct,"np":np,"trapezoid":TRAPEZOID}
    exec(programs[0],namespace)
    return namespace


def source_calculation(programs):
    namespace=new_source_state(programs)
    exec(programs[1],namespace)
    exec(programs[2],namespace)
    states=namespace["states"]
    return {
        "time":states.t.copy(),"crank_angle":states.ca.copy(),"temperature":states.T.copy(),
        "pressure":states.P.copy(),"volume":states.V.copy(),"mass":states.m.copy(),
        "entropy_mass":states.entropy_mass.copy(),"mean_molecular_weight":states.mean_molecular_weight.copy(),
        "mdot_in":states.mdot_in.copy(),"mdot_out":states.mdot_out.copy(),
        "work_rate":states.dWv_dt.copy(),"heat_release_rate":(states.heat_release_rate*states.V).copy(),
        "X":states.X.copy(),"Y":states.Y.copy(),
        "integrals":dict(heat_J=float(namespace["Q"]),work_J=float(namespace["W"]),
                         efficiency=float(namespace["eta"]),CO_ppm=float(1e6*namespace["CO_emission"])),
        "solver_stats":namespace["sim"].solver_stats,
        "rtol":namespace["sim"].rtol,"atol":namespace["sim"].atol,
    }


def output_replay(actual,expected):
    return all(np.array_equal(actual[key],expected[key]) for key in expected
               if isinstance(expected[key],np.ndarray)) and actual["integrals"]==expected["integrals"] \
        and actual["solver_stats"]==expected["solver_stats"] and actual["rtol"]==expected["rtol"] \
        and actual["atol"]==expected["atol"]


def source_checks(result):
    return (result["rtol"]==1e-12 and result["atol"]==1e-16 and len(result["time"])>2880
        and .16 <= result["time"][-1] < .16+1/(360*50)
        and all(np.isfinite(value).all() for value in result.values() if isinstance(value,np.ndarray))
        and np.all(np.diff(result["time"])>0) and np.min(result["Y"])>=-1e-12)


def source_snapshot(namespace,time_offset=0.):
    gas=namespace["cyl"].phase
    cylinder=namespace["cyl"]
    t=namespace["sim"].time+time_offset
    work=-(gas.P-namespace["ambient_air"].phase.P)*namespace["A_piston"]*namespace["piston_speed"](t)
    return dict(time=t,temperature=gas.T,pressure=gas.P,volume=cylinder.volume,mass=cylinder.mass,
        entropy_mass=gas.entropy_mass,mean_molecular_weight=gas.mean_molecular_weight,
        mdot_in=namespace["inlet_valve"].mass_flow_rate,mdot_out=namespace["outlet_valve"].mass_flow_rate,
        mdot_fuel=namespace["injector_mfc"].mass_flow_rate,work_rate=work,
        heat_release_rate=gas.heat_release_rate*cylinder.volume,CO_X=gas["co"].X[0])


def reference_segments(namespace):
    """Independent tight reference on continuous, local-time source segments."""
    network=namespace["sim"]
    network.rtol,network.atol=REFINED_RTOL,REFINED_ATOL
    network.max_steps=1000000
    network.max_time_step=1/(360*50)
    stops=np.unique(np.r_[0.,[(720*cycle+angle)/(360*50) for cycle in range(4)
        for angle in (18,198,350,365,522,702)],.16])
    for start,stop in zip(stops[:-1],stops[1:]):
        midpoint=(start+stop)/2
        for device,opening,delta in (("inlet_valve","inlet_open","inlet_delta"),
                ("outlet_valve","outlet_open","outlet_delta"),
                ("injector_mfc","injector_open","injector_delta")):
            value=np.mod(namespace["crank_angle"](midpoint)-namespace[opening],4*np.pi)<namespace[delta]
            namespace[device].time_function=lambda t,value=value:value
        namespace["piston"].velocity=lambda t,start=start:namespace["piston_speed"](t+start)
        network.initial_time=0.
        network.initialize()
        yield float(start),float(stop)


def refined_reference(programs,times):
    namespace=new_source_state(programs)
    network=namespace["sim"]
    rows,mass_fractions=[],[]
    for start,stop in reference_segments(namespace):
        selected=times[(times>start)&(times<=stop)]
        if start==0:
            rows.append(source_snapshot(namespace))
            mass_fractions.append(namespace["cyl"].phase.Y.copy())
        for t in selected:
            network.advance(float(t-start),apply_limit=False)
            rows.append(source_snapshot(namespace,start))
            mass_fractions.append(namespace["cyl"].phase.Y.copy())
        network.advance(stop-start,apply_limit=False)
    result={key:np.array([row[key] for row in rows]) for key in rows[0]}
    result["Y"]=np.asarray(mass_fractions)
    return result


def integral_terms(output,indices=None):
    ix=np.arange(len(output["time"])) if indices is None else indices
    t=output["time"][ix]
    heat=TRAPEZOID(output["heat_release_rate"][ix],t)
    work=TRAPEZOID(output["work_rate"][ix],t)
    weights=output["mean_molecular_weight"][ix]*output["mdot_out"][ix]
    return np.array([heat,work,TRAPEZOID(weights*output["CO_X"][ix],t),TRAPEZOID(weights,t)])


def integral_values(terms):
    heat,work,numerator,denominator=terms
    return dict(heat_J=float(heat),work_J=float(work),efficiency=float(work/heat),CO_ppm=float(1e6*numerator/denominator))


def refined_integrals(programs):
    # Preserve the published ODE tolerances and resolve its quadrature on every
    # accepted step. The independent pointwise reference above separately uses
    # tighter ODE tolerances and exact continuous-regime restarts.
    namespace=new_source_state(programs)
    network=namespace["sim"]
    network.max_time_step=1/(360*50)
    network.initialize()
    rows=[source_snapshot(namespace)]
    while network.time<.16:
        if .16-network.time<=1/(360*50):
            # End exactly at eight revolutions, retaining dense observations
            # of the final interval instead of one long quadrature panel.
            for t in np.linspace(network.time,.16,17)[1:]:
                network.advance(float(t),apply_limit=False)
                rows.append(source_snapshot(namespace))
        else:
            network.step()
            rows.append(source_snapshot(namespace))
    output={key:np.array([row[key] for row in rows]) for key in rows[0]}
    selected=np.unique(np.r_[np.arange(0,len(rows),2),len(rows)-1])
    integral=integral_values(integral_terms(output))
    coarse=integral_values(integral_terms(output,selected))
    convergence={key:abs(coarse[key]/value-1) for key,value in integral.items()}
    return output,integral,convergence


def compare_native(native,summary,reference,reference_integrals):
    actual={key[len("output_"):]:native[key] for key in native.files if key.startswith("output_")}
    errors={
        "temperature_K":float(np.max(np.abs(actual["temperature"]-reference["temperature"]))),
        "pressure_relative":float(np.max(np.abs(actual["pressure"]/reference["pressure"]-1))),
        "volume_m3":float(np.max(np.abs(actual["volume"]-reference["volume"]))),
        "mass_relative":float(np.max(np.abs(actual["mass"]/reference["mass"]-1))),
        "mass_fraction":float(np.max(np.abs(actual["Y"]-reference["Y"]))),
        "entropy_J_per_kg_K":float(np.max(np.abs(actual["entropy_mass"]-reference["entropy_mass"]))),
    }
    limits=dict(temperature_K=.5,pressure_relative=2e-4,volume_m3=2e-10,mass_relative=5e-5,
                mass_fraction=1e-4,entropy_J_per_kg_K=.5)
    switches=np.array([(720*cycle+angle)/(360*50) for cycle in range(4)
        for angle in (18,198,350,365,522,702)])
    continuous=np.min(np.abs(actual["time"][:,None]-switches),axis=1)>1e-11
    rate_errors={key:float(np.max(np.abs(actual[key][continuous]-reference[key][continuous]))/
        max(np.max(np.abs(reference[key][continuous])),1e-30))
        for key in ("mdot_in","mdot_out","mdot_fuel","work_rate","heat_release_rate")}
    rate_limits=dict(mdot_in=2e-3,mdot_out=2e-3,mdot_fuel=1e-13,work_rate=2e-3,heat_release_rate=5e-3)
    integral_errors={key:abs(summary["integrals"][key]/value-1) for key,value in reference_integrals.items()}
    integral_limits=dict(heat_J=1e-4,work_J=1e-5,efficiency=1e-4,CO_ppm=1e-4)
    return dict(correctness_pass=all(errors[k]<v for k,v in limits.items()) and
        all(integral_errors[k]<v for k,v in integral_limits.items()) and
        all(rate_errors[k]<v for k,v in rate_limits.items()),trajectory_errors=errors,
        trajectory_limits=limits,rate_peak_scaled_errors=rate_errors,rate_limits=rate_limits,
        integral_relative_errors=integral_errors,integral_limits=integral_limits)


def verify_mechanisms(source,native_path,native_meta):
    if sha(native_path)!=native_meta["mechanism_sha256"] or sha(str(native_path)+".npz")!=native_meta["sidecar_sha256"]:
        raise ValueError("native mechanism/sidecar files differ from the timed Julia artifact")
    sidecar_meta=decode_metadata(np.load(str(native_path)+".npz"))
    if sidecar_meta.get("source_sha256")!=sha(native_path):
        raise ValueError("native sidecar provenance does not match its YAML")
    original,prepared=ct.Solution(str(source),"nDodecane_IG"),ct.Solution(str(native_path))
    if original.species_names!=prepared.species_names or original.n_species!=100 or not original.n_reactions==prepared.n_reactions==553:
        raise ValueError("source/prepared phases do not contain the required 100 species and 553 reactions")
    for T in (300.,1000.,2500.):
        original.TPX=prepared.TPX=T,1.3e5,"o2:1,n2:3.76"
        for key in ("molecular_weights","standard_enthalpies_RT","standard_cp_R","standard_entropies_R",
                    "forward_rate_constants","reverse_rate_constants"):
            np.testing.assert_allclose(getattr(original,key),getattr(prepared,key),rtol=2e-12,atol=1e-20,
                                       err_msg=f"prepared phase differs in {key} at {T} K")
    return sidecar_meta


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--julia-result",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--native-mechanism",type=Path,required=True)
    parser.add_argument("--source-example",type=Path,required=True)
    parser.add_argument("--source-mechanism",type=Path)
    parser.add_argument("--cantera-build-record",type=Path)
    parser.add_argument("--repetitions",type=int,default=9)
    parser.add_argument("--qualification",choices=("informational","controlled"),default="informational")
    parser.add_argument("--target",choices=("wsl","apple-m4"),default="wsl")
    parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args()
    if args.repetitions<MIN_REPETITIONS:
        parser.error("at least nine warm repetitions required")
    native=np.load(args.julia_result)
    require_native_replay_flags(native)
    meta=decode_metadata(native)
    summary=tomllib.loads(meta["summary_toml"])
    if meta["scope"]!="full_source_eight_revolutions":
        raise ValueError("full eight-revolution native calculation required")
    mechanism=args.source_mechanism or next((Path(d)/"nDodecane_Reitz.yaml" for d in ct.get_data_directories()
        if (Path(d)/"nDodecane_Reitz.yaml").is_file()),None)
    if mechanism is None:
        raise FileNotFoundError("provide --source-mechanism")
    sidecar_meta=verify_mechanisms(mechanism,args.native_mechanism,meta)
    programs=source_programs(args.source_example,mechanism)
    hardware=host_metadata()
    hardware["load_average_start"]=os.getloadavg()
    thread_checks={"before_first":verify_numerical_threads(set_accelerate=True)}
    started=time.perf_counter()
    first=source_calculation(programs)
    first_seconds=time.perf_counter()-started
    first_checked=require_source_checks(first,"first invocation")
    thread_checks["before_warm"]=verify_numerical_threads(set_accelerate=True)
    print("Cantera source first calculation:",first_seconds,"s; checks",first_checked,flush=True)
    samples,matches,checks=[],[],[]
    if not args.validate_only:
        gc.collect()
        for repetition in range(args.repetitions):
            started=time.perf_counter()
            repeated=source_calculation(programs)
            samples.append(time.perf_counter()-started)
            thread_checks[f"after_warm_{repetition+1}"]=verify_numerical_threads()
            matches.append(require_source_replay(repeated,first,f"warm repetition {repetition+1}"))
            checks.append(require_source_checks(repeated,f"warm repetition {repetition+1}"))
            print("Cantera source warm repetition",repetition+1,samples[-1],"s; checks",matches[-1] and checks[-1],flush=True)
    started=time.perf_counter()
    reference=refined_reference(programs,native["output_time"])
    integral_output,integrals,convergence=refined_integrals(programs)
    validation_seconds=time.perf_counter()-started
    thread_checks["after_all"]=verify_numerical_threads()
    accuracy=compare_native(native,summary,reference,integrals)
    reference_converged=max(convergence.values())<1e-4
    reference_path=args.output.with_suffix(".reference.npz")
    np.savez(reference_path,**{"pointwise_"+k:v for k,v in reference.items()},
             **{"accepted_"+k:v for k,v in integral_output.items()},
             **{"source_"+k:v for k,v in first.items() if isinstance(v,np.ndarray)})
    record=json.loads(args.cantera_build_record.read_text()) if args.cantera_build_record else None
    source_commit=record["source"]["commit"] if record else getattr(ct,"__git_commit__","unknown")
    libraries=cantera_library_hashes(ct.__file__)
    extension_sha=sha(compiled.__file__)
    recorded={Path(k).name:v for k,v in (record or {}).get("library_hashes",{}).items()}
    build_matches=bool(libraries and recorded and all(recorded.get(k)==v for k,v in libraries.items())
        and recorded.get(Path(compiled.__file__).name)==extension_sha)
    thread_keys=("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")+(("VECLIB_MAXIMUM_THREADS",) if hardware["system"]=="Darwin" else ())
    threads={key:os.environ.get(key) for key in thread_keys}
    native_threads=tomllib.loads(meta["thread_environment_toml"])
    native_thread_checks=tomllib.loads(meta.get("thread_checks_toml",""))
    native_host={key:meta.get(key,"") for key in ("cpu","system","kernel_release")}
    native_samples=native["warm_seconds"].tolist()
    native_matches=native["warm_matches_first"].astype(bool).tolist()
    native_checks=native["warm_checked"].astype(bool).tolist()
    source_hashes=tomllib.loads(meta["source_hashes_toml"])
    root=Path(__file__).resolve().parents[1]
    code_matches=all((root/path).is_file() and sha(root/path)==digest for path,digest in source_hashes.items())
    native_thread_helper_matches=meta.get("thread_helper_sha256")==sha(root/"validation"/"numerical_threads.jl")
    expected_native_checks={"before_first","before_warm","after_all"}|{f"after_warm_{i+1}" for i in range(len(native_samples))}
    native_actual_threads_valid=(set(native_thread_checks)==expected_native_checks and
        all(values and all(value==1 for value in values.values()) for values in native_thread_checks.values()))
    all_repetitions_checked=(len(samples)==len(matches)==len(checks)>=MIN_REPETITIONS and
        len(native_samples)==len(native_matches)==len(native_checks)>=MIN_REPETITIONS and
        all(matches) and all(checks) and all(native_matches) and all(native_checks))
    correct=bool(accuracy["correctness_pass"] and reference_converged and first_checked
        and all(matches) and all(checks) and all(native_matches) and all(native_checks)
        and native["first_checked"][0] and native["source_hashes_unchanged"][0])
    controlled=bool(not args.validate_only and args.qualification==meta["qualification"]=="controlled"
        and matches_target(hardware,args.target) and matches_target(native_host,args.target)
        and all(hardware[k]==native_host[k] for k in ("cpu","kernel_release"))
        and ct.__version__.startswith("4.0") and source_commit==SOURCE_COMMIT and build_matches and code_matches
        and int(native["julia_threads"][0])==int(native["blas_threads"][0])==1
        and all(v=="1" for v in threads.values()) and all(v=="1" for v in native_threads.values())
        and native_actual_threads_valid and native_thread_helper_matches
        and all_repetitions_checked)
    ratio=statistics.median(samples)/statistics.median(native_samples) if samples and native_samples else None
    report=dict(example="reactors/ic_engine.py",scope=meta["scope"],benchmark_target=args.target,
        numerical_thread_checks=thread_checks,native_numerical_thread_checks=native_thread_checks,
        native_actual_threads_valid=native_actual_threads_valid,native_thread_helper_matches=native_thread_helper_matches,
        python_thread_helper_sha256=sha(root/"validation"/"benchmark_environment.py"),
        source_commit=SOURCE_COMMIT,source_example_sha256=sha(args.source_example),
        source_mechanism_sha256=sha(mechanism),native_mechanism_sha256=sha(args.native_mechanism),
        native_sidecar_sha256=sha(str(args.native_mechanism)+".npz"),sidecar_provenance=sidecar_meta,
        cantera_version=ct.__version__,cantera_source_sha=source_commit,
        cantera_build_record_sha256=sha(args.cantera_build_record) if record else None,
        cantera_shared_libraries_sha256=libraries,cantera_extension_sha256=extension_sha,
        loaded_libraries_match_build_record=build_matches,harness_sha256=sha(__file__),
        native_artifact_sha256=sha(args.julia_result),reference_artifact_sha256=sha(reference_path),
        native_metadata=meta,native_source_hashes=source_hashes,native_code_matches_current_files=code_matches,
        hardware=dict(hardware,load_average_end=os.getloadavg()),thread_settings=threads,
        native_thread_settings=native_threads,timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        timed_scope="fresh mechanism/network construction, complete eight-revolution solve, output properties and integrals; imports, plotting, printing, reference validation and artifact I/O excluded",
        native_scope_note="actual shared solve_ic_engine, ic_engine_observables and ic_engine_summary calls, including converged accepted-state heat/CO quadrature and pressure-work ledger",
        source_scope_note="AST-selected original pinned calculation with its SolutionArray output, default tolerances, 20 K advance limit, one-degree requests and sampled integral estimates",
        qualification_timing_baseline="source-default Cantera calculation; refined reference is excluded from speed ratio",
        refined_reference_note="pointwise reference: tighter ODE tolerances with local-time restarts at exact switches; integral reference: published ODE tolerances with every accepted state and a separate sampling-convergence check",
        cantera_tolerances=dict(rtol=1e-12,atol=1e-16),refined_tolerances=dict(rtol=REFINED_RTOL,atol=REFINED_ATOL),
        integral_reference_tolerances=dict(rtol=1e-12,atol=1e-16),
        native_tolerances=dict(rtol=1e-13,species_atol_kg=1e-26,temperature_atol_K=1e-10,volume_atol_m3=1e-20),
        cantera_import_seconds=IMPORT_SECONDS,julia_import_seconds=float(native["import_seconds"][0]),
        cantera_first_seconds=first_seconds,julia_first_seconds=float(native["first_seconds"][0]),
        first_call_note="full first invocation in each process; Julia includes JIT; all warm runs create fresh states",
        cantera_warm_seconds=samples,julia_warm_seconds=native_samples,
        cantera_warm_matches_first=matches,cantera_warm_checked=checks,
        native_warm_matches_first=native_matches,native_warm_checked=native_checks,
        all_repetitions_checked=all_repetitions_checked,source_solver_stats=first["solver_stats"],
        source_output_points=len(first["time"]),source_end_time_s=float(first["time"][-1]),
        source_maximum_output_temperature_change_K=float(np.max(np.abs(np.diff(first["temperature"])))),
        source_maximum_output_interval_s=float(np.max(np.diff(first["time"]))),
        mechanism_species=100,mechanism_reactions=553,
        source_sampled_integrals=first["integrals"],native_summary=summary,
        refined_integrals=integrals,refined_integral_points=len(integral_output["time"]),
        refined_quadrature_change=convergence,refined_quadrature_converged=reference_converged,
        accuracy=accuracy,correctness_pass=correct,reference_validation_seconds=validation_seconds,
        speed_ratio=ratio,minimum_speed_ratio=.95,speed_ratio_definition="median Cantera seconds / median Julia seconds",
        qualification="controlled" if controlled else "not_qualified",
        performance_pass=bool(controlled and correct and ratio is not None and ratio>=.95))
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("Engine correctness",correct,"speed ratio",ratio,"controlled",controlled,flush=True)
    if not correct:
        raise SystemExit("native engine failed independent refined-reference checks")


if __name__=="__main__":
    main()

"""Time complete source-default flame sequences, with separate physical references."""
import argparse,hashlib,json,os,platform,statistics,subprocess,sys,time
from pathlib import Path
for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","JULIA_NUM_THREADS"]:os.environ[key]="1"
import numpy as np
import cantera as ct
from benchmark_environment import host_metadata,matches_target,loaded_library_paths,verify_numerical_threads

PINNED_COMMIT="726522be4e2a13454d8415b7ef799d621f665cf3"
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def loaded_libraries():
    return [str(path) for path in loaded_library_paths()]


def verify_threads(*,set_accelerate=False):
    result=verify_numerical_threads(set_accelerate=set_accelerate)
    result["environment"]={key:os.environ[key] for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","JULIA_NUM_THREADS"]}
    return result

def cantera_sequence(gas,case,profile,save_profiles=False):
    free=case=="free";fixed=case=="fixed"
    modes=["mass","mass-soret","multi","multi-soret"] if free else ["mole","multi"]
    seconds=[];points=[];snapshots={}
    # Restore the preloaded gas's initial transport manager, corresponding to
    # fresh mechanism loading outside the timed source calculation.
    gas.transport_model="mixture-averaged"
    for stage,mode in enumerate(modes):
        multi=mode.startswith("multi");start=time.perf_counter()
        if stage==0:
            gas.TPX=(300.,ct.one_atm,"H2:1.1,O2:1,AR:5") if free else (373.7,ct.one_atm,"CH4:.65,O2:1,N2:3.76") if fixed else (373.,.05*ct.one_atm,"H2:1.5,O2:1,AR:7")
            flame=ct.FreeFlame(gas,width=.03) if free else ct.BurnerFlame(gas,width=.01 if fixed else .5)
            if not free:flame.burner.mdot=.04 if fixed else .06
            if fixed:
                flame.flame.set_fixed_temp_profile(profile["positions"]/.01,profile["temperatures"])
                flame.energy_enabled=False
        flame.transport_model="multicomponent" if multi else "mixture-averaged"
        flame.flux_gradient_basis="mass" if free else "molar"
        flame.soret_enabled=mode.endswith("soret")
        slope=.06 if free else (.1 if multi else .3) if fixed else .05
        curve=.12 if free else (.2 if multi else 1.) if fixed else .1
        flame.set_refine_criteria(ratio=3.,slope=slope,curve=curve)
        # Preserve the published default tolerances and Jacobian mode here.
        # Refined accuracy-reference solves are entirely outside this runner.
        flame.solve(loglevel=0,auto=stage==0 and not fixed)
        seconds.append(time.perf_counter()-start);points.append(len(flame.grid))
        if save_profiles:
            snapshots[mode]=dict(grid=flame.grid.copy(),T=flame.T.copy(),Y=flame.Y.copy(),velocity=flame.velocity.copy(),
                inlet_Y=flame.inlet.Y.copy() if free else flame.burner.Y.copy(),P=[flame.P])
    return seconds,points,snapshots

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("parameters",type=Path);p.add_argument("references",type=Path);p.add_argument("output",type=Path)
    p.add_argument("case",choices=["free","burner","fixed"])
    p.add_argument("--project",required=True,type=Path);p.add_argument("--julia",required=True)
    p.add_argument("--build-record",required=True,type=Path)
    p.add_argument("--native-record",required=True,type=Path)
    p.add_argument("--target",required=True,choices=["wsl","apple-m4"])
    p.add_argument("--reps",type=int,default=9);p.add_argument("--formal",action="store_true")
    p.add_argument("--order",choices=["cantera-first","julia-first"],default="cantera-first")
    a=p.parse_args()
    if a.reps<5:p.error("at least five warm repetitions required")
    if not ct.__version__.startswith("4.0"):p.error("Cantera4 required")
    build=json.loads(a.build_record.read_text())
    if build.get("source",{}).get("commit")!=PINNED_COMMIT:p.error("build record does not identify pinned pristine source")
    native_record=json.loads(a.native_record.read_text())
    source_hashes={str(path.relative_to(a.project)):digest(path) for path in sorted((a.project/"src").rglob("*.jl"))}
    if source_hashes!=native_record.get("source_hashes"):p.error("native source files do not match the pinned checkpoint record")
    host=host_metadata()
    if not matches_target(host,a.target):p.error(f"host mismatch: {host}")
    a.output.mkdir(parents=True,exist_ok=True)
    native=a.output/"native";native.mkdir(exist_ok=True)
    cantera=a.output/"cantera";cantera.mkdir(exist_ok=True)
    mechanism=a.parameters/("gri30.yaml" if a.case=="fixed" else "h2o2.yaml")
    gas=ct.Solution(str(mechanism))
    if (gas.n_species,gas.n_reactions)!=((53,325) if a.case=="fixed" else (10,29)):p.error("exact stock mechanism required")
    profile=np.load(a.parameters/"fixed-profile.npz") if a.case=="fixed" else None
    threads=verify_threads(set_accelerate=True)
    report=dict(case=a.case,formal=a.formal,passed=False,performance_pass=False,date=time.strftime("%Y-%m-%d %H:%M:%S %z"),host=host,target=a.target,
        scope="Sum of construction/initialization/adaptive-solve stages in the complete published transport sequence. Mechanism/sidecar loading, snapshots and output excluded. No refined-reference solve is timed.",
        compilation_scope="Julia runtime startup and using/imports are outside timers. Specialization of run_source_sequence before entry to its internal stage timers is also excluded; the first measured sequence is not whole-program cold latency. Only JIT triggered after a stage timer starts can enter its measurement. First repetition is recorded separately and excluded from warm medians.",
        timing_order=a.order,threads=threads,cantera_version=ct.__version__,cantera_build_record=build,
        cantera_build_record_sha256=digest(a.build_record),mechanism_sha256=digest(mechanism),
        sidecar_sha256=digest(str(mechanism)+".npz"),multicomponent_sha256=digest(str(mechanism)+".multicomponent.npz"),
        fixed_temperature_profile_sha256=digest(a.parameters/"fixed-profile.npz") if a.case=="fixed" else None,
        parameter_provenance=json.loads((a.parameters/"provenance.json").read_text()),
        reference_hashes={path.name:digest(path) for path in sorted(a.references.glob(a.case+"-*.npz"))},
        benchmark_driver_hashes={path.name:digest(path) for path in [Path(__file__),Path(__file__).with_suffix(".jl"),
            Path(__file__).with_name("conservative_source_accuracy.py"),Path(__file__).with_name("flame_benchmarks.py"),
            Path(__file__).with_name("benchmark_environment.py")]},
        native_source_hashes=source_hashes,native_commit=native_record.get("commit"),native_record_sha256=digest(a.native_record))
    def run_ct():
        times=[];nodes=[];baseline=None;replay=[]
        for repetition in range(a.reps+1):
            elapsed,points,snapshots=cantera_sequence(gas,a.case,profile,True)
            times.append(elapsed);nodes.append(points)
            if baseline is None:baseline=snapshots
            for mode,data in snapshots.items():
                errors={};hasher=hashlib.sha256()
                for key in sorted(data):
                    value=np.asarray(data[key]);initial=np.asarray(baseline[mode][key])
                    if value.shape!=initial.shape or not np.all(np.isfinite(value)) or not np.allclose(value,initial,rtol=1e-12,atol=1e-14):
                        raise RuntimeError(f"Cantera numerical replay failed: {repetition} {mode} {key}")
                    errors[key]=float(np.max(np.abs(value-initial)))
                    hasher.update(key.encode());hasher.update(str(value.shape).encode());hasher.update(value.tobytes())
                replay.append(dict(repetition=repetition,mode=mode,sha256=hasher.hexdigest(),max_abs=errors))
                if repetition==0:np.savez(cantera/f"{a.case}-{mode}-first.npz",**data)
            print("cantera",a.case,repetition,sum(elapsed),elapsed,points,flush=True)
            if repetition==a.reps:
                for mode,data in snapshots.items():np.savez(cantera/f"{a.case}-{mode}-0.npz",**data)
        report["cantera_stage_seconds"]=times;report["cantera_stage_points"]=nodes
        report["cantera_replay"]=dict(passed=True,checked_stages=len(replay),relative_tolerance=1e-12,absolute_tolerance=1e-14,records=replay)
    def run_julia():
        command=[a.julia,f"--project={a.project}",str(Path(__file__).with_suffix(".jl")),str(a.parameters),str(native),a.case,str(a.reps)]
        result=subprocess.run(command,capture_output=True,text=True,env=dict(os.environ))
        (a.output/"julia.log").write_text(result.stdout+result.stderr)
        print(result.stdout,end="",flush=True)
        if result.returncode:raise RuntimeError(f"Julia failed: {result.stderr[-3000:]}")
        data=np.load(native/"timings.npz")
        if int(data["julia_threads"][0])!=1 or int(data["blas_threads"][0])!=1:raise RuntimeError("Julia thread guard failed")
        if platform.system()=="Darwin" and int(data["accelerate_threading"][0])!=1:raise RuntimeError("Julia Accelerate thread guard failed")
        text=lambda key:bytes(data[key+"_utf8"]).decode()
        if text("kernel")!=host["kernel_release"]:raise RuntimeError("Julia and Cantera host/kernel mismatch")
        if Path(text("package_path")).resolve()!=(a.project/"src/Arrhenius.jl").resolve():raise RuntimeError("Julia loaded a different Arrhenius checkout")
        report["julia_stage_seconds"]=data["stage_seconds"].tolist();report["julia_stage_points"]=data["stage_points"].tolist()
        report["julia_runtime"]={key:text(key) for key in ["julia_version","kernel","machine","package_path","blas_config","loaded_libraries"]}
        report["julia_runtime"]["accelerate_threading"]=int(data["accelerate_threading"][0])
        expected=(a.reps+1)*(4 if a.case=="free" else 2)
        if int(data["replay_checked_stages"][0])!=expected:raise RuntimeError("incomplete Julia replay checks")
        report["julia_replay"]=dict(passed=True,checked_stages=expected,relative_tolerance=1e-12,absolute_tolerance=1e-14,
            hashes=text("replay_hashes").splitlines(),fields=text("replay_fields").splitlines(),max_abs=data["replay_max_abs"].tolist())
        report["julia_blas_library_hashes"]={path:digest(path) for path in text("loaded_libraries").splitlines()
            if any(word in Path(path).name.lower() for word in ["blas","lapack"]) and Path(path).is_file()}
    try:
        for run in ([run_ct,run_julia] if a.order=="cantera-first" else [run_julia,run_ct]):run()
        report["threads_after"]=verify_threads()
        import cantera._cantera as compiled
        paths={Path(path).resolve() for path in loaded_libraries() if "cantera" in Path(path).name.lower() and Path(path).is_file()}
        paths.add(Path(compiled.__file__).resolve())
        hashes={str(path):digest(path) for path in sorted(paths)}
        if len(hashes)<2 or any(value not in build["library_hashes"].values() for value in hashes.values()):raise RuntimeError(f"loaded Cantera library/build-record mismatch: {hashes}")
        report["actual_loaded_cantera_hashes"]=hashes
        checker=Path(__file__).with_name("conservative_source_accuracy.py")
        accuracy=a.output/"accuracy.json"
        result=subprocess.run([sys.executable,str(checker),str(mechanism),str(native),str(a.references),a.case,str(accuracy)],capture_output=True,text=True)
        print(result.stdout,end="",flush=True)
        report["accuracy"]=json.loads(accuracy.read_text()) if accuracy.is_file() else dict(passed=False,error=result.stderr)
        ct_times=np.sum(report["cantera_stage_seconds"],axis=1);jl_times=np.sum(report["julia_stage_seconds"],axis=1)
        report["cantera_seconds"]=ct_times.tolist();report["julia_seconds"]=jl_times.tolist()
        ratio=float(statistics.median(ct_times[1:])/statistics.median(jl_times[1:]))
        report["median_speed_ratio_cantera_over_julia"]=ratio
        physical=result.returncode==0 and report["accuracy"]["passed"]
        replay=report["cantera_replay"]["passed"] and report["julia_replay"]["passed"]
        report["ratio_threshold_met"]=ratio>=.95
        report["physical_gate_application"]="Last repetition of every source stage checked against refined references; strict numerical replay establishes the same outputs for all first/warm repetitions."
        report["performance_pass"]=bool(a.formal and physical and replay and ratio>=.95)
        report["passed"]=bool(physical and replay and (not a.formal or report["performance_pass"]))
    except Exception as error:
        report["failure"]=str(error)
    (a.output/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({key:report.get(key) for key in ["case","passed","median_speed_ratio_cantera_over_julia","failure"]}),flush=True)
    return 0 if report["passed"] else 1
if __name__=="__main__":raise SystemExit(main())

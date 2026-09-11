# Complete published transport sequences; no reference states enter a solve.
using Arrhenius,LinearAlgebra,NPZ,Libdl,SHA
BLAS.set_num_threads(1)
accelerate_threading=-1
if Sys.isapple()
    accelerate=Libdl.dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate")
    ccall(Libdl.dlsym(accelerate,:BLASSetThreading),Cint,(Cuint,),1)==0 || error("Accelerate thread setting failed")
    global accelerate_threading=Int(ccall(Libdl.dlsym(accelerate,:BLASGetThreading),Cuint,()))
    accelerate_threading==1 || error("Accelerate is not single-threaded")
end
parameters,output,case=ARGS[1:3]
reps=parse(Int,ARGS[4]);reps>=5 || error("at least five warm repetitions required")
Threads.nthreads()==1 && BLAS.get_num_threads()==1 || error("single-thread benchmark required")
mkpath(output)
mechanism=joinpath(parameters,case=="fixed" ? "gri30.yaml" : "h2o2.yaml")
gas=CreateSolution(mechanism);data=MultiTransportData(mechanism*".multicomponent.npz",gas;mechanism)
profile=case=="fixed" ? npzread(joinpath(parameters,"fixed-profile.npz")) : nothing
function run_source_sequence(gas,data,case,profile;save_profiles=false)
    free=case=="free";fixed=case=="fixed"
    modes=free ? ["mass","mass-soret","multi","multi-soret"] : ["mole","multi"]
    seconds=zeros(length(modes));points=zeros(Int,length(modes));snapshots=Dict{String,Any}()
    f=nothing
    for (stage,mode) in enumerate(modes)
        multi=startswith(mode,"multi")
        seconds[stage]=@elapsed begin
            if stage==1
                X=free ? "H2:1.1,O2:1,AR:5" : fixed ? "CH4:.65,O2:1,N2:3.76" : "H2:1.5,O2:1,AR:7"
                f=free ? FreeFlame(gas;T=300.,P=one_atm,X,width=.03) :
                    BurnerFlame(gas;T=fixed ? 373.7 : 373.,P=fixed ? one_atm : .05one_atm,X,
                        mdot=fixed ? .04 : .06,width=fixed ? .01 : .5)
                f.discretization==:conservative || error("benchmark requires the conservative default")
                if fixed
                    set_temperature_profile!(f,vec(profile["positions"]),vec(profile["temperatures"]);relative=false)
                end
            end
            set_transport!(f,multi ? :multicomponent : :mixture_averaged;data,soret=endswith(mode,"soret"),
                flux_gradient_basis=free ? :mass : :mole)
            slope=free ? .06 : fixed ? (multi ? .1 : .3) : .05
            curve=free ? .12 : fixed ? (multi ? .2 : 1.) : .1
            solve!(f;ratio=3.,slope,curve)
        end
        f.converged || error("$case $mode failed")
        points[stage]=length(f.grid)
        if save_profiles
            snapshots[mode]=Dict("grid"=>copy(f.grid),"T"=>temperature(f),"Y"=>mass_fractions(f),
                "velocity"=>velocity(f),"inlet_Y"=>copy(f.inlet_Y),"state"=>copy(f.state),"P"=>[f.pressure])
        end
    end
    return seconds,points,snapshots
end
modes=case=="free" ? ["mass","mass-soret","multi","multi-soret"] : ["mole","multi"]
fields=["grid","T","Y","velocity","inlet_Y","state","P"]
seconds=zeros(reps+1,length(modes));points=zeros(Int,size(seconds));snapshots=nothing;first_profiles=nothing
replay_errors=zeros(reps+1,length(modes),length(fields));hashes=String[]
for repetition in 0:reps
    elapsed,nodes,saved=run_source_sequence(gas,data,case,profile;save_profiles=true)
    seconds[repetition+1,:].=elapsed;points[repetition+1,:].=nodes
    repetition==0 && (global first_profiles=saved)
    for (stage,mode) in enumerate(modes)
        buffer=IOBuffer()
        for (field,key) in enumerate(fields)
            current,baseline=saved[mode][key],first_profiles[mode][key]
            size(current)==size(baseline) || error("replay dimensions changed for $case $mode $key")
            all(isfinite,current) || error("nonfinite replay output")
            all(abs.(current.-baseline) .<= 1e-14 .+ 1e-12.*abs.(baseline)) || error("numerical replay failed for $case $mode $key")
            replay_errors[repetition+1,stage,field]=maximum(abs,current.-baseline)
            write(buffer,key);write(buffer,Int64.(collect(size(current))));write(buffer,vec(current))
        end
        push!(hashes,bytes2hex(SHA.sha256(take!(buffer))))
        repetition==0 && npzwrite(joinpath(output,"$case-$mode-first.npz"),saved[mode])
    end
    repetition==reps && (global snapshots=saved)
    println((;case,repetition,seconds=sum(elapsed),stages=elapsed,points=nodes));flush(stdout)
end
for (mode,values) in snapshots
    npzwrite(joinpath(output,"$case-$mode-0.npz"),values)
end
npzwrite(joinpath(output,"timings.npz"),Dict("stage_seconds"=>seconds,"stage_points"=>points,
    "replay_checked_stages"=>[(reps+1)*length(modes)],"replay_max_abs"=>replay_errors,
    "replay_hashes_utf8"=>collect(codeunits(join(hashes,"\n"))),"replay_fields_utf8"=>collect(codeunits(join(fields,"\n"))),
    "julia_version_utf8"=>collect(codeunits(string(VERSION))),"julia_threads"=>[Threads.nthreads()],"blas_threads"=>[BLAS.get_num_threads()],
    "accelerate_threading"=>[accelerate_threading],
    "kernel_utf8"=>collect(codeunits(readchomp(`uname -r`))),"machine_utf8"=>collect(codeunits(string(Sys.MACHINE))),
    "package_path_utf8"=>collect(codeunits(pathof(Arrhenius))),
    "blas_config_utf8"=>collect(codeunits(string(BLAS.get_config()))),
    "loaded_libraries_utf8"=>collect(codeunits(join(Libdl.dllist(),"\n")))))

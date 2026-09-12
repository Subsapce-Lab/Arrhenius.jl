# Loader for electron-collision cross-section data in Cantera-style plasma YAML.
# Tables are used verbatim: energy levels in eV, cross sections in m^2 as written
# in the source file (no top-level unit conversion is applied here).

const _EEDF_KINDS = Dict{String,ElectronCollisionKind}(
    "effective" => EffectiveCollision,
    "elastic" => ElasticCollision,
    "excitation" => ExcitationCollision,
    "ionization" => IonizationCollision,
    "attachment" => AttachmentCollision,
)

_eedf_error(msg) = throw(ArgumentError(msg))

function _eedf_float(x, what)
    x isa Real || _eedf_error("$what must be a real number, got $(repr(x))")
    v = Float64(x)
    isfinite(v) || _eedf_error("$what must be finite")
    return v
end

function _eedf_table(entry, ctx)
    e = get(entry, "energy-levels", nothing)
    s = get(entry, "cross-sections", nothing)
    e isa AbstractVector || _eedf_error("$ctx: missing or invalid energy-levels table")
    s isa AbstractVector || _eedf_error("$ctx: missing or invalid cross-sections table")
    length(e) == length(s) ||
        _eedf_error("$ctx: energy-levels and cross-sections length mismatch " *
                    "($(length(e)) vs $(length(s)))")
    length(e) >= 2 || _eedf_error("$ctx: at least two cross-section table points required")
    energy = [_eedf_float(v, "$ctx energy level") for v in e]
    cross = [_eedf_float(v, "$ctx cross section") for v in s]
    any(<(0.0), energy) && _eedf_error("$ctx: energies must be nonnegative")
    any(<(0.0), cross) && _eedf_error("$ctx: cross sections must be nonnegative")
    all(>(0.0), diff(energy)) ||
        _eedf_error("$ctx: energy levels must be strictly increasing")
    return energy, cross
end

function _eedf_threshold(entry, ctx)
    thr = _eedf_float(get(entry, "threshold", 0.0), "$ctx threshold")
    thr < 0.0 && _eedf_error("$ctx: threshold must be nonnegative")
    return thr
end

# Parse one side of a reaction equation into species => stoichiometric count.
# Splits on the spaced " + " separator so ionic names such as "O2+" survive.
function _eedf_side(side, ctx)
    counts = Dict{String,Int}()
    for term in split(side, " + "; keepempty=false)
        term = strip(term)
        isempty(term) && continue
        m = match(r"^(\d+)\s+(.+)$", term)
        name = m === nothing ? term : strip(m.captures[2])
        n = m === nothing ? 1 : parse(Int, m.captures[1])
        isempty(name) && _eedf_error("$ctx: empty species name in '$side'")
        counts[name] = get(counts, name, 0) + n
    end
    isempty(counts) && _eedf_error("$ctx: no species found in '$side'")
    return counts
end

function _eedf_reaction_kind(react, prod)
    react == prod && return ElasticCollision
    re = get(react, "Electron", 0)
    pe = get(prod, "Electron", 0)
    pe > re && return IonizationCollision
    pe < re && return AttachmentCollision
    return ExcitationCollision
end

function _eedf_grid(phase, ctx)
    eedf = get(phase, "electron-energy-distribution", nothing)
    eedf isa AbstractDict ||
        _eedf_error("$ctx: phase has no electron-energy-distribution")
    type = get(eedf, "type", nothing)
    type == "Boltzmann-two-term" ||
        _eedf_error("$ctx: unsupported electron-energy-distribution type " *
                    "$(repr(type)); only Boltzmann-two-term is supported")
    levels = get(eedf, "energy-levels", nothing)
    levels isa AbstractVector ||
        _eedf_error("$ctx: missing or invalid EEDF energy-levels")
    edges = [_eedf_float(v, "$ctx EEDF energy level") for v in levels]
    length(edges) >= 3 ||
        _eedf_error("$ctx: EEDF grid needs at least 3 energy levels")
    any(<(0.0), edges) &&
        _eedf_error("$ctx: EEDF energy levels must be nonnegative")
    all(>(0.0), diff(edges)) ||
        _eedf_error("$ctx: EEDF energy levels must be strictly increasing")
    return edges
end

function _eedf_phase(root, phase)
    phases = get(root, "phases", nothing)
    phases isa AbstractVector && !isempty(phases) ||
        _eedf_error("file has no phases section")
    if phase === nothing
        idx = findfirst(p -> p isa AbstractDict &&
                             haskey(p, "electron-energy-distribution"), phases)
        idx === nothing &&
            _eedf_error("no phase declares an electron-energy-distribution; " *
                        "pass phase=<name> to select one explicitly")
        return phases[idx]
    end
    idx = findfirst(p -> p isa AbstractDict && get(p, "name", nothing) == phase,
                    phases)
    idx === nothing && _eedf_error("phase $(repr(phase)) not found")
    return phases[idx]
end

"""
    read_eedf_model(path; phase=nothing) -> EEDFModel

Read electron-collision cross sections from a Cantera-style plasma YAML file.
Only `Boltzmann-two-term` EEDF definitions are supported. Root-level
`electron-collisions` entries require an explicit `kind`; reactions of type
`electron-collision-plasma` get their kind from electron stoichiometry and must
have exactly one collision target besides `Electron`. A zero threshold is kept as
zero for root entries; for plasma reactions it is replaced by the first
strictly positive tabulated energy level.
"""
function read_eedf_model(path; phase=nothing)
    root = YAML.load_file(path)
    root isa AbstractDict || _eedf_error("$(path): not a YAML mapping")
    edges = _eedf_grid(_eedf_phase(root, phase), "read_eedf_model")

    collisions = ElectronCollision[]
    targets = String[]
    seen_background = Set{String}()

    function add!(target, kind, thr, energy, cross, origin)
        if kind === EffectiveCollision || kind === ElasticCollision
            key = target
            key in seen_background &&
                _eedf_error("duplicate $(kind) record for target '$target'")
            push!(seen_background, key)
        end
        target in targets || push!(targets, target)
        push!(collisions,
              ElectronCollision(target, kind, thr, energy, cross, origin))
    end

    entries = get(root, "electron-collisions", Any[])
    entries isa AbstractVector ||
        _eedf_error("electron-collisions must be a list")
    for entry in entries
        entry isa AbstractDict || _eedf_error("electron-collisions: entry must be a mapping")
        ctx = "electron-collisions entry $(repr(get(entry, "target", nothing)))"
        target = get(entry, "target", nothing)
        target isa AbstractString && !isempty(strip(target)) && target != "Electron" ||
            _eedf_error("electron-collisions entry: missing target")
        ks = get(entry, "kind", nothing)
        haskey(_EEDF_KINDS, ks) ||
            _eedf_error("$ctx: explicit kind required, one of " *
                        "$(join(sort(collect(keys(_EEDF_KINDS))), ", "))")
        energy, cross = _eedf_table(entry, ctx)
        add!(String(target), _EEDF_KINDS[ks], _eedf_threshold(entry, ctx),
             energy, cross, :root)
    end

    reactions = get(root, "reactions", Any[])
    reactions isa AbstractVector || _eedf_error("reactions must be a list")
    for r in reactions
        r isa AbstractDict || _eedf_error("reactions: entry must be a mapping")
        get(r, "type", nothing) == "electron-collision-plasma" || continue
        eq = get(r, "equation", nothing)
        eq isa AbstractString ||
            _eedf_error("electron-collision-plasma reaction: missing equation")
        ctx = "reaction '$eq'"
        occursin("<=>", eq) && _eedf_error("$ctx: reversible electron collisions are unsupported")
        sides = split(eq, "=>")
        length(sides) == 2 || _eedf_error("$ctx: malformed equation")
        react = _eedf_side(sides[1], ctx)
        prod = _eedf_side(sides[2], ctx)
        get(react, "Electron", 0) == 1 || _eedf_error("$ctx: one incident Electron required")
        neutral = filter(!=("Electron"), collect(keys(react)))
        length(neutral) == 1 ||
            _eedf_error("$ctx: expected exactly one collision target besides " *
                        "Electron, got $(join(neutral, ", "))")
        react[neutral[1]] == 1 || _eedf_error("$ctx: one target particle required")
        energy, cross = _eedf_table(r, ctx)
        thr = _eedf_threshold(r, ctx)
        if thr == 0.0
            idx = findfirst(>(0.0), energy)
            idx === nothing &&
                _eedf_error("$ctx: cannot infer threshold; no positive energy level")
            thr = energy[idx]
        end
        add!(neutral[1], _eedf_reaction_kind(react, prod), thr,
             energy, cross, :reaction)
    end

    isempty(targets) &&
        _eedf_error("$(path): no electron collisions found; no participating targets")
    return EEDFModel(edges, collisions, targets)
end

using Arrhenius, LinearAlgebra, Test, YAML

function _pathomech_fixture(edit, base, dir, name)
    root = YAML.load_file(base)
    edit(root)
    path = joinpath(dir, "$name.yaml")
    YAML.write_file(path, root)
    return path
end

function _pathomech_reaction(root, equation)
    only(filter(r -> get(r, "equation", "") == equation, root["reactions"]))
end

@testset "pathological mechanism diagnostics and finite evaluation" begin
    base = joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml")
    mktempdir() do dir
        cases = Dict{String,String}()
        for (name, delta) in (("nasa7_small", 1e-3), ("nasa7_large", 1.0))
            cases[name] = _pathomech_fixture(base, dir, name) do root
                h2 = only(filter(s -> s["name"] == "H2", root["species"]))
                h2["thermo"]["data"][2][1] += delta
            end
        end
        cases["plog_underflow"] = _pathomech_fixture(base, dir, "plog_underflow") do root
            r = _pathomech_reaction(root, "O + H2 <=> H + OH")
            rc = r["rate-constant"]
            delete!(r, "rate-constant")
            r["type"] = "pressure-dependent-Arrhenius"
            r["rate-constants"] = [
                Dict("P"=>"0.1 atm", "A"=>rc["A"], "b"=>rc["b"], "Ea"=>1e6),
                Dict("P"=>"10 atm", "A"=>rc["A"], "b"=>rc["b"], "Ea"=>1e6)]
        end
        cases["duplicate_unflagged"] = _pathomech_fixture(base, dir, "duplicate_unflagged") do root
            r = deepcopy(_pathomech_reaction(root, "O + H2 <=> H + OH"))
            r["id"] = "duplicate-candidate"
            push!(root["reactions"], r)
        end
        cases["unbalanced"] = _pathomech_fixture(base, dir, "unbalanced") do root
            _pathomech_reaction(root, "O + H2 <=> H + OH")["equation"] = "H2 + O <=> H + H"
        end
        cases["negative_A"] = _pathomech_fixture(base, dir, "negative_A") do root
            _pathomech_reaction(root, "O + H2 <=> H + OH")["rate-constant"]["A"] *= -1
        end
        cases["extreme_Ea"] = _pathomech_fixture(base, dir, "extreme_Ea") do root
            _pathomech_reaction(root, "O + H2 <=> H + OH")["rate-constant"]["Ea"] = 1e7
        end
        cases["collision_limit"] = _pathomech_fixture(base, dir, "collision_limit") do root
            _pathomech_reaction(root, "O + H2 <=> H + OH")["rate-constant"]["A"] = 1e15
        end

        expected = Dict(
            "nasa7_small" => :nasa_continuity,
            "nasa7_large" => :nasa_continuity,
            "plog_underflow" => :plog_underflow_risk,
            "duplicate_unflagged" => :duplicate_unflagged_candidates,
            "unbalanced" => :element_balance,
            "negative_A" => :negative_A,
            "collision_limit" => :collision_limit_screen)
        for (name, path) in cases
            report = audit_mechanism(path)
            if haskey(expected, name)
                @test any(x -> x.check == expected[name], report.issues)
            end
            gas = CreateSolution(path)
            Y = ones(gas.n_species) ./ gas.n_species
            for T in (250.0, 999.999, 1000.0, 1000.001, 2500.0, 3500.0)
                source = set_states(gas, T, one_atm, Y)
                @test all(isfinite, source)
                X = Y2X(gas, Y, 1 / dot(Y, 1 ./ gas.MW))
                @test all(isfinite, Arrhenius.cal_cp_R(gas, T, one_atm, X))
                @test all(isfinite, Arrhenius.cal_h_RT(gas, T, one_atm, X))
                @test all(isfinite, Arrhenius.cal_s0_R(gas, T, one_atm, X))
            end
        end
    end
end

using SparseArrays

# Independent copy of the pre-fusion formula, including output conversion.
function equilibrium_reference!(work, vk, vk_sum, reversible, temperature, entropy, enthalpy)
    gas_constant = oftype(temperature, R)
    one_atmosphere = oftype(temperature, one_atm)
    mul!(work.delta_s, transpose(vk), entropy)
    mul!(work.delta_h, transpose(vk), enthalpy)
    for i in eachindex(work.kf)
        work.equilibrium_constants[i] = exp(
            work.delta_s[i] / gas_constant -
            work.delta_h[i] / (gas_constant * temperature) +
            log(one_atmosphere / gas_constant / temperature) * vk_sum[i])
        work.kr[i] = reversible[i] ? work.kf[i] / work.equilibrium_constants[i] :
            zero(temperature)
    end
end

function equilibrium_test_workspace(::Type{T}, nr) where {T}
    # Preserve the exported six-vector constructor.
    return KineticsWorkspace(ntuple(_ -> zeros(T, nr), 6)...)
end

@testset "fused equilibrium rates" begin
    update! = Arrhenius._equilibrium_rates!
    for Scalar in (Float32, Float64), ns in (2, 7), nr in (0, 1, 11)
        dense = [Scalar(mod(i + 2j, 5) - 2) / 2 for i in 1:ns, j in 1:nr]
        if nr > 1
            dense[:, 2] .= 0 # Includes an empty reaction column.
        end
        vk = sparse(dense)
        sums = vec(sum(vk; dims=1))
        reversible = [isodd(i) for i in 1:nr]
        entropy = Scalar.(range(-1000, 1500; length=ns))
        for temperature in Scalar.((400, 1200, 2800)), displacement in Scalar.((0, 1e-4, 0.3))
            # Near equilibrium is deliberately cancellation-sensitive.
            enthalpy = temperature .* entropy .+ displacement .* Scalar.(1:ns)
            work = equilibrium_test_workspace(Scalar, nr)
            work.kf .= Scalar.(1:nr)
            reference = deepcopy(work)
            entropy_before, enthalpy_before = copy(entropy), copy(enthalpy)
            equilibrium_reference!(reference, vk, sums, reversible, temperature, entropy, enthalpy)
            @test update!(work, vk, sums, reversible, temperature, entropy, enthalpy) === nothing
            for field in (:delta_s, :delta_h, :equilibrium_constants, :kr)
                @test getfield(work, field) ≈ getfield(reference, field) rtol=16eps(Scalar)
            end
            @test entropy == entropy_before
            @test enthalpy == enthalpy_before
            @test work.kf == reference.kf
            @test @allocated(update!(work, vk, sums, reversible, temperature, entropy, enthalpy)) == 0
        end
    end

    # Stored zeros and value/structure mutations are read on every call.
    vk = sparse([1, 2, 3], [1, 1, 2], [-1.0, 1.0, 0.0], 3, 2)
    entropy, enthalpy = [300.0, 700.0, -100.0], [4e5, 8e5, 1e5]
    work = equilibrium_test_workspace(Float64, 2)
    work.kf .= [1.0, 3.0]
    for mutation in 1:4
        if mutation == 2
            vk[1, 1] = -0.5
        elseif mutation == 3
            vk[2, 2] = 0.75
        elseif mutation == 4
            dropzeros!(vk)
        end
        sums = vec(sum(vk; dims=1))
        reference = deepcopy(work)
        equilibrium_reference!(reference, vk, sums, [true, false], 1000.0, entropy, enthalpy)
        update!(work, vk, sums, [true, false], 1000.0, entropy, enthalpy)
        @test work.equilibrium_constants ≈ reference.equilibrium_constants
        @test work.kr ≈ reference.kr
    end

    # Dense, mixed-precision and caller-owned workspaces retain the generic path.
    for matrix in (vk, Matrix(vk)), Scalar in (Float32, Float64)
        for caller_owned in (false, true)
            storage = equilibrium_test_workspace(Scalar, 2)
            storage.kf .= [1, 3]
            target = caller_owned ? NamedTuple{fieldnames(typeof(storage))}(
                Tuple(getfield(storage, name) for name in fieldnames(typeof(storage)))) : storage
            reference = deepcopy(storage)
            sums = vec(sum(matrix; dims=1))
            equilibrium_reference!(reference, matrix, sums, [true, false], Scalar(1000), entropy, enthalpy)
            update!(target, matrix, sums, [true, false], Scalar(1000), entropy, enthalpy)
            @test target.equilibrium_constants ≈ reference.equilibrium_constants
            @test target.kr ≈ reference.kr
        end
    end

    function temperature_response(temperature, reference=false)
        scalar = typeof(temperature)
        work = equilibrium_test_workspace(scalar, 2)
        work.kf .= [1, 3]
        s = scalar.(entropy)
        h = temperature .* s .+ scalar.([1, 2, 3])
        updater = reference ? equilibrium_reference! : update!
        updater(work, vk, vec(sum(vk; dims=1)), [true, true], temperature, s, h)
        return sum(work.kr)
    end
    @test ForwardDiff.derivative(temperature_response, 1000.0) ≈
        ForwardDiff.derivative(t -> temperature_response(t, true), 1000.0)
    @test ForwardDiff.derivative(t -> ForwardDiff.derivative(temperature_response, t), 1000.0) ≈
        ForwardDiff.derivative(t -> ForwardDiff.derivative(u -> temperature_response(u, true), t), 1000.0)
    @test_throws DimensionMismatch update!(work, vk, [0.0, 0.0], [true, true], 1000.0,
        entropy[1:2], enthalpy)
end

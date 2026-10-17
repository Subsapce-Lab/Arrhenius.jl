using Arrhenius, LinearAlgebra, Test

@testset "premixed Newton correction allocation" begin
    B = 2
    n = 4
    bandwidth = 2B - 1
    band = zeros(6B - 2,n)
    band[4B-1,:] .= 1
    _,pivots = LinearAlgebra.LAPACK.gbtrf!(bandwidth,bandwidth,n,band)
    residual = reshape([1.0,-2.0,3.0,-4.0],B,2)
    correction = zeros(n)
    step = reshape(Arrhenius._flame_solve_correction!(correction,band,pivots,residual,bandwidth),B,2)
    @test step == -residual
    Arrhenius._flame_solve_correction!(correction,band,pivots,residual,bandwidth)
    @test (@allocated Arrhenius._flame_solve_correction!(correction,band,pivots,residual,bandwidth)) == 0
end

@testset "premixed residual and Jacobian allocation" begin
    mechanism = joinpath(@__DIR__,"..","mechanism","h2o2.yaml")
    gas = CreateSolution(mechanism)
    X = mole_fractions(gas,"H2:1.1,O2:1,AR:5")
    flame = solve!(FreeFlame(gas;X,discretization=:finite_difference))
    workspace = Arrhenius.FlameWorkspace(flame)
    residual = similar(flame.state)
    Arrhenius.flame_residual!(residual,flame,flame.state,workspace)
    @test (@allocated Arrhenius.flame_residual!(residual,flame,flame.state,workspace)) == 0
    Arrhenius._flame_jacobian(flame,flame.state,workspace,residual;analytic=false)
    @test (@allocated Arrhenius._flame_jacobian(flame,flame.state,workspace,residual;analytic=false)) == 0
end

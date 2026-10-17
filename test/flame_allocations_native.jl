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
    step = Arrhenius._flame_solve_correction!(correction,band,pivots,residual,bandwidth)
    @test step == -residual
    Arrhenius._flame_solve_correction!(correction,band,pivots,residual,bandwidth)
    @test (@allocated Arrhenius._flame_solve_correction!(correction,band,pivots,residual,bandwidth)) == 0
end

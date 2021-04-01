using JMPS,LinearAlgebra,OMEinsum
using Test
using CUDA
CUDA.allowscalar(false)

@testset "Constructor" begin
    mps = MPS(5,2,4,FloatType=Float64,ArrayType=Array)
    for i = 1:5
        @test  norm(mps[i]) ≈ 0
    end
    mpo = MPO(5,2,4,FloatType=ComplexF64,ArrayType=Array)
    for i = 1:5
        @test  norm(mpo[i]) ≈ 0
    end
    mps = MPS(5,2,4,FloatType=Float64,ArrayType=Array)    
    for i = 1:5
        @test  norm(mps[i]) ≈ 0
    end
    mpo = MPO(5,2,4,FloatType=ComplexF64,ArrayType=Array)
    for i = 1:5
        @test  norm(mpo[i]) ≈ 0
    end
    mps = MPS(5,2,4,FloatType=Float64,ArrayType=CuArray)    
    for i = 1:5
        @test  norm(mps[i]) ≈ 0
    end
    mpo = MPO(5,2,4,FloatType=ComplexF64,ArrayType=CuArray)
    for i = 1:5
        @test  norm(mpo[i]) ≈ 0
    end
    mps = MPS(5,2,4,FloatType=Float64,ArrayType=CuArray)    
    for i = 1:5
        @test  norm(mps[i]) ≈ 0
    end
    mpo = MPO(5,2,4,FloatType=ComplexF64,ArrayType=CuArray)
    for i = 1:5
        @test  norm(mpo[i]) ≈ 0
    end
end

@testset "get-set index" begin
    mps = randomMPS(5,2,4,FloatType=Float64,ArrayType=Array)
    mps[3] .= 0 
    @test norm(mps[3])≈0
    mpo = randomMPO(3,3,3,FloatType=Float64,ArrayType=Array)
    mpo[2] .= 0 
    @test norm(mpo[2])≈0
end

@testset "Bdims" begin
    bdim = JMPS.Bdims(4,[2;3;4;5])
    @test bdim[0] == 5
    @test bdim[1] == 2
    @test bdim[2] == 3
    @test bdim[3] == 4
    @test bdim[4] == 5
    @test bdim[5] == 2
end

@testset "single_tensor_multiply" begin
    mpo = rand(4,5,6,7)
    mps = rand(3,7,4)
    @test norm(JMPS.single_tensor_multiply(mpo,mps)-ein"lurd,adb->laurb"(mpo, mps))≈0
end

@testset "Adjoint MPS" begin
    mps1 = randomMPS(4,4,31)
    mps2 = randomMPS(4,4,31)
    @test (mps1*mps2-mps2*mps1)≈0
end

@testset "normalization and overlap" begin
    mps = randomMPS(4,4,51)
    normalization!(mps,LeftNormalization())
    mps1 = deepcopy(mps)
    mps2 = mps
    compress!(mps2,50)
    ovlp = mps2*mps1
    n1 = mps1*mps1
    n2 = mps2*mps2
    @test n1 - n2 ≈ 0 atol = 3E-7
    @test ovlp - n2 ≈ 0 atol = 3E-7
    @test ovlp - n1 ≈ 0 atol = 3E-7
end


@testset "N=2 AKLT chain" begin
    N = 2
    σ⁺ = [0 1;0 0 ]
    σ⁻ = [0 0;1 0 ]
    σᶻ = [1 0;0 -1]
    mps = MPS(N,3,2)
    mps[1][1,1,:] .= ( sqrt(2/3)σ⁺)[1,:]
    mps[1][1,2,:] .= (-sqrt(1/3)σᶻ)[1,:]
    mps[1][1,3,:] .= (-sqrt(2/3)σ⁻)[1,:]
    mps[N][:,1,1] .= ( sqrt(2/3)σ⁺)[:,1]
    mps[N][:,2,1] .= (-sqrt(1/3)σᶻ)[:,1]
    mps[N][:,3,1] .= (-sqrt(2/3)σ⁻)[:,1]
    @test disp(mps,[1;3])≈-2/3
    @test disp(mps,[3;1])≈0
    @test disp(mps,[2;2])≈1/3
    @test exp(mps*mps) ≈ 5/9
    x = 1/5
    @test entropy(mps,1) ≈ -(x*log(x)+4x*log(4x)) #will normalize mps
    @test sum(entropy(mps)) ≈ -(x*log(x)+4x*log(4x))
    normalization!(mps,LeftNormalization())
    @test disp(mps,[1;3])≈-2/sqrt(5)
    @test disp(mps,[2;2])≈1/sqrt(5)
end

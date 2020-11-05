using JMPS
using Test

@testset "JMPS_BASIC.jl" begin
mps = MPS(5,2,4,FloatType=Float64)
@test sum(mps[1])==0
@test sum(abs.(mps[5]))==0

L = 16 
S = 2
res = 0.0
mps = MPS(L,3,31,FloatType=Float64)
for i = 1:1:L
    mps[i] = rand(mps.bdim[i-1],mps.S,mps.bdim[i])
end
mps_old = deepcopy(mps)
res = compress!(mps, 30)
ovlp_old = overlap(mps_old, mps)
ovlp = overlap(mps, mps_old)

normalization!(mps,LeftNormalization())
@test overlap(mps,mps) ≈ 1.0 atol = 1E-5
normalization!(mps_old,RightNormalization())
@test overlap(mps,mps_old) ≈ 1.0 atol = 1E-5

a = SciNum(1.0,1.0)
@test (ovlp - ovlp_old) ≈ 0.0 atol = 1E-5  #check of multiply and compress
print("Test of overlap and compress PASS\n")

mps2 = MPS(L,3,15,FloatType=Float64)
mpo = MPO(L,3,5,FloatType=Float64)
for i = 1:1:L
    mps2[i] = rand(mps2.bdim[i-1],mps2.S,mps2.bdim[i])
    mpo[i] = rand(mpo.bdim[i-1],mpo.S,mpo.bdim[i],mpo.S)
end
mps1 = mps
t = mps1*mpo
result1 = (mps1*mpo)*mps2
result2 = mps1*(mpo*mps2)
@test Float64(result1-result2) ≈ 0.0 atol = 1E-10
print("Test of multiply PASS\n")

mps2 = MPS(L,3,15,FloatType=ComplexF64)
mpo = MPO(L,3,5,FloatType=ComplexF64)
for i = 1:1:L
    mps2[i] = rand(mps2.bdim[i-1],mps2.S,mps2.bdim[i])
    mpo[i] = rand(mpo.bdim[i-1],mpo.S,mpo.bdim[i],mpo.S)
end
mps1 = mps
t = mps1*mpo
result1 = (mps1*mpo)*mps2
result2 = mps1*(mpo*mps2)
@test result1-result2 ≈ 0.0 atol = 1E-10
@test sum(entropy(mps1))>0
@test sum(entropy(mps2))>0
compress!(mps2,2)
@test sum(entropy(mps2))>0
print("Test of ComplexF64 PASS\n")
end

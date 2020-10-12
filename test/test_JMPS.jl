using JMPS
using Test

@testset "JMPS_BASIC.jl" begin
mps = MPS(Float64,5,2,4)
@test sum(mps[1])==0
@test sum(abs.(mps[5]))==0

L = 16 
S = 2
res = 0.0
mps = MPS(Float64,L,3,11)
for i = 1:1:L
    mps[i] = rand(mps.bdim[i-1],mps.S,mps.bdim[i])
end
mps_old = deepcopy(mps)
res = compress!(mps, 10)
ovlp_old = overlap(mps_old, mps)
ovlp = overlap(mps, mps_old)

a = SciNum(1.0,1.0)
@test (ovlp - ovlp_old) ≈ 0.0 atol = 1E-10  #check of multiply and compress
print("Test of overlap and compress PASS\n")

mps2 = MPS(Float64,L,3,15)
mpo = MPO(Float64,L,3,5)
for i = 1:1:L
    mps2[i] = rand(mps2.bdim[i-1],mps2.S,mps2.bdim[i])
    mpo[i] = rand(mpo.bdim[i-1],mpo.S,mpo.bdim[i],mpo.S)
end
mps1 = mps
t = mps1*mpo
result1 = (mps1*mpo)*mps2
result2 = mps1*(mpo*mps2)
@test result1-result2 ≈ 0.0 atol = 1E-10
print("Test of multiply PASS\n")

mps2 = MPS(ComplexF64,L,3,15)
mpo = MPO(ComplexF64,L,3,5)
for i = 1:1:L
    mps2[i] = rand(mps2.bdim[i-1],mps2.S,mps2.bdim[i])
    mpo[i] = rand(mpo.bdim[i-1],mpo.S,mpo.bdim[i],mpo.S)
end
mps1 = mps
t = mps1*mpo
result1 = (mps1*mpo)*mps2
result2 = mps1*(mpo*mps2)
@test result1-result2 ≈ 0.0 atol = 1E-10
@tesst sum(entropy(mps1))>0
print("Test of ComplexF64 PASS\n")
end

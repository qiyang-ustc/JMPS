struct MPO{FloatType<:Number,ArrayType<:AbstractArray{FloatType,4}} <: abstractMPO
    L::Int #length
    S::Int #physical index
    bdim::Bdims
    tensor::Array{ArrayType,1}
end

function MPO(L::Int,S::Int,D::Int;FloatType=Float64,ArrayType=Array) 
    bdim = Bdims(L,append!([D for i=1:L-1],1))
    tensor = [zeros(FloatType,bdim[i-1],S,bdim[i],S) for i=1:L]
    return MPO{FloatType,ArrayType{FloatType,4}}(L,S,bdim,tensor)
end

"""
    transpose for an operator
"""
function transpose!(mpo::MPO)
    for i = 1:1:mpo.L
        mpo[i] = permutedims(mpo[i],[1,4,3,2])
    end
    return mpo
end
transpose(mpo::MPO) = transpose!(deepcopy(mpo))

function multiply(mpo::MPO,mps::MPS)
    temp_mps = deepcopy(mps)
    multiply!(mpo,temp_mps)
    return temp_mps
end

"""
for an MPO => T, multiply(mpo,mps)= T|mps>
"""
multiply!(mpo::MPO,mps::MPS) = multiply!(mpo,mps,1,mps.L)

"""
Multiply only from site s to site e
Please check s,e in 1:mps.L
"""
function multiply!(mpo::MPO,mps::MPS,s::Int,e::Int) 
    for site = (s>e ? (e:s) : (s:e))
        temp_site = single_tensor_multiply(mpo[site],mps[site]) # mps[site] = ein"lurd,adb->laurb"(mpo[site], mps[site]) 
        mps.bdim[site-1] = mpo.bdim[site-1]*mps.bdim[site-1] # only change left 
        mps[site] = reshape(temp_site,(mps.bdim[site-1], mps.S,:))
    end
    return mps
end

*(mps1::MPS,mps2::MPS) = overlap(dag(mps1),mps2)
*(mpo::MPO,mps::MPS) = multiply(mpo,mps)
*(mps::MPS,mpo::MPO) = multiply(transpose(mpo),mps)

function cuda(mpo::MPO{FloatType,Array{FloatType,4}} where FloatType<:Number)
    return MPO{FloatType,CuArray{FloatType,4}}(mpo.L,mpo.S,mpo.bdim,map(CuArray,mpo.tensor))
end
import CUDA.cu
export cu

function cu(mpo::MPO{FloatType,Array{FloatType,4}}) where FloatType<:Number
    return MPO{FloatType,CuArray{FloatType,4}}(mpo.L,mpo.S,mpo.bdim,map(CuArray,mpo.tensor))
end

function cu(mps::MPS{FloatType,Array{FloatType,3}}) where FloatType<:Number
    return MPS{FloatType,CuArray{FloatType,3}}(mps.L,mps.S,mps.bdim,map(CuArray,mps.tensor))
end
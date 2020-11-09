import CUDA.cu
export cu,array

function cu(mpo::MPO{FloatType,Array{FloatType,4}}) where FloatType<:Number
    return MPO{FloatType,CuArray{FloatType,4}}(mpo.L,mpo.S,mpo.bdim,map(CuArray,mpo.tensor))
end

function cuda(mps::MPO{FloatType,Array{FloatType,3}} where FloatType<:Number)
    return MPS{FloatType,CuArray{FloatType,3}}(mps.L,mps.S,mps.bdim,map(CuArray,mps.tensor))
end

function array(mpo::MPO{FloatType,CuArray{FloatType,4}} where FloatType<:Number)
    return MPO{FloatType,Array{FloatType,4}}(mpo.L,mpo.S,mpo.bdim,map(Array,mpo.tensor))
end
function array(mps::MPO{FloatType,CuArray{FloatType,3}} where FloatType<:Number)
    return MPS{FloatType,Array{FloatType,3}}(mps.L,mps.S,mps.bdim,map(Array,mps.tensor))
end

"""
MPS is a series of tensors, and also a vector in Hilbert space.
"""
mutable struct MPS{FloatType<:Number,ArrayType<:AbstractArray{FloatType,3}} <: AbstractMPS
    L::Int #length
    S::Int #physical index
    bdim::Bdims #bond dimension
    tensor::Array{ArrayType,1}
end

function MPS(L::Int,S::Int,D::Int;FloatType=Float64,ArrayType=Array)
    bdim = Bdims(L,append!([D for i=1:L-1],1))
    tensor = [zeros(FloatType,bdim[i-1],S,bdim[i]) for i=1:L]
    return MPS{FloatType,ArrayType{FloatType,3}}(L,S,bdim,tensor)
end

"""
overlap of two mps: mps1 and mps2.
the same as transpose(mps2)*mps1

Automatically using CUDA
"""
overlap(mps1::AbstractMPS,mps2::AbstractMPS;ArraType=CuArray) = _overlap(mps1,mps2,ArraType)

"""
    Compress a MPS to bond dimension: Dcut
"""
function compress!(mps::AbstractMPS,Dcut::Int,epsilon=1E-13;use_cuda=false)
    normalization!(mps,LeftNormalization())
    res = cutoff!(mps,Dcut,LeftNormalization())
    return res
end

function dag!(mps::MPS{T,V}) where T<:Complex where V
    for i=1:1:mps.L
        mps[i] = conj.(mps[i])
    end
    return mps
end

function dag(mps::MPS{T,V}) where T<:Complex where V
    temp = deepcopy(mps)
    dag!(temp)
    return temp
end

dag(mps::MPS{T,V}) where T<: Real where V = mps
dag!(mps::MPS{T,V}) where T<: Real where V = mps

"""
    display mps(array...)
    Just think that it display an element for a vector in Hilbert space.
"""
function disp(mps::AbstractMPS,array::Array{Int,1})
    @assert mps.L == size(array)[1]
    t = transpose(mps[1][1,array[1],:])
    for i = 2:mps.L
        t = t*mps[i][:,array[i],:]
    end
    return t[1,1]
end
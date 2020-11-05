"""
MPS is a series of tensors, and also a vector in Hilbert space.
"""
mutable struct MPS{FloatType<:Number,ArrayType<:AbstractArray{FloatType,3}} <: abstractMPS
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
"""
function overlap(mps1::abstractMPS,mps2::abstractMPS)
    epsilon = 1E-13
    E = transpose(reshape(mps1[1],(mps1.S, mps1.bdim[1]))) * reshape(mps2[1],(mps2.S, mps2.bdim[1]))
    scale = SciNum(1.0)
    for i=2:mps1.L
        #E = torch.einsum('ab,ade,bdf->ef', (E, mps1[i], mps2[i])) # einsum version of following three lines
        a=size(mps1[i])[1]
        e=size(mps1[i])[3]
        E = transpose(reshape(transpose(E) * reshape(mps1[i],a,:),:,e))
        E = E * reshape(mps2[i],:,mps2.bdim[i])
        s = norm(E)
        if abs(s)< epsilon
            return SciNum(s)
        end
        scale = scale*SciNum(s)
        E = E./s # devided by norm
    end
    temp = sum(E)
    @assert abs(imag(temp))<1E-10 # TODO: we do not calculate non-real overlap. May need complex support
    return SciNum(real(temp))*scale
end

"""
    Compress a MPS to bond dimension: Dcut
"""
function compress!(mps::abstractMPS,Dcut::Int,epsilon=1E-13)
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
function disp(mps::abstractMPS,array::Array{Int,1})
    @assert mps.L == size(array)[1]
    t = transpose(mps[1][1,array[1],:])
    for i = 2:mps.L
        t = t*mps[i][:,array[i],:]
    end
    return t
end


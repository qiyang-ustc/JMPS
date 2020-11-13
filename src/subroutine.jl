# This file contain lowerest level-1 function
# this is a patch for CUDA.allowscalar(false)
USE_CUDA = false
export allowcuda
function allowcuda(b::Bool)
    global USE_CUDA = b
end

import Base.convert
convert(::Type{CuArray{T,2}},U::CUDA.CUSOLVER.CuQRPackedQ{T,CuArray{T,2}}) where T<: Number = CuArray(U)
convert(::Type{Array{T,2}}, U::LinearAlgebra.QRCompactWYQ{T,Array{T,2}}) where T<: Number = Array(U)

"""
    see multiply!
"""
function single_tensor_multiply(mpo_tensor::AbstractArray{T,4} where T<:Number,mps_tensor::AbstractArray{T,3} where T<:Number)
    (l,u,r,d)=size(mpo_tensor)
    (a,d,b)=size(mps_tensor)
    if USE_CUDA
        mpo_tensor = CuArray(mpo_tensor)
        mps_tensor = CuArray(mps_tensor)
    end
    temp_site = reshape(mpo_tensor,:,d)*reshape(permutedims(mps_tensor,(2,1,3)),d,:) #lurab
    temp_site = permutedims(reshape(temp_site,(l,u,r,a,b)),(1,4,2,3,5))
    return Array(temp_site)
end

"""
    sweep in a single site
"""
function single_tensor_sweep!(mps::AbstractMPS,site::Int,::LeftNormalization)
    l=mps.bdim[site-1] # left bond dimension
    r=mps.bdim[site]   # current bond dimension
    A=reshape(mps[site],(l*mps.S,r)) # A is a matrix unfolded from the current tensor
    if USE_CUDA
        A = CuArray(A)
    end
    U,R = qr!(A) # here we intent to do QR = A. However there is no BP, so we do SVD instead 
    A = nothing # clear cache
    s = norm(R)
    R = R./s # devided by norm
    res = log(s)
    U = convert(typeof(R),U) 
    Dnew = size(R)[1]

    U = reshape(U ,l,mps.S,Dnew)
    mps[site] = Array(U)   # U is LinearAlgebra.QRCompactWYQ type,it is not normal array
    temp = reshape(mps[site+1],r,:)
    if USE_CUDA
        temp = CuArray(temp)
    end
    mps[site+1] = Array(reshape(R*temp,:,mps.S,mps.bdim[site+1]))
    mps.bdim[site] = Dnew
    return res
end

function single_tensor_sweep!(mps::AbstractMPS,site::Int,::RightNormalization)
    l = mps.bdim[site-1]
    r = mps.bdim[site]
    A = copy(reshape(mps[site],(l, r*mps.S)))
    if USE_CUDA
        A = CuArray(A)
    end
    A = copy(transpose(A))
    U,R = qr!(A) # UR = A^{T} R^{T}U^{T} = A
    A = nothing
    Dnew = size(R)[1]
    U = convert(typeof(R),U)
    U = convert(typeof(R),transpose(U))
    R = convert(typeof(U),transpose(R))
    mps[site] = Array(reshape(U,(Dnew,mps.S,r)))
    temp = reshape(mps[site-1],mps.bdim[site-2]*mps.S,mps.bdim[site-1])
    if USE_CUDA
        temp = CuArray(temp)
    end
    mps[site-1] = Array(reshape(temp* R ,(mps.bdim[site-2], mps.S, Dnew)))
    mps.bdim[site-1] = Dnew
end

"""
    cut-off a single tensor
"""
function single_tensor_cutoff!(mps::AbstractMPS,site::Int,Dcut::Int,::LeftNormalization;epsilon=0)
    l = mps.bdim[site-1]
    r = mps.bdim[site]
    A = reshape(mps[site],(l, r*mps.S))
    if USE_CUDA
        A = CuArray(A)
    end
    U, S, V = svd!(A)
    A = nothing
    Dnew = min(Dcut, sum(S.>epsilon))
    res = sum(S[Dnew+1:min(l,r*mps.S)])
    V = copy(adjoint(V))[1:Dnew,:]
    V = reshape(V,(Dnew,mps.S,:))
    mps[site] = Array(V)
    temp = reshape(mps[site-1],mps.bdim[site-2]*mps.S,mps.bdim[site-1])
    if USE_CUDA
        temp = CuArray(temp)
    end
    mps[site-1] = Array(reshape(temp* U[:,1:Dnew] *Diagonal(S[1:Dnew]),(mps.bdim[site-2], mps.S, Dnew)))
    mps.bdim[site-1] = Dnew
    return res
end

"""
    find the sigular value spectrum for a single tensor
"""
function single_tensor_spectrum!(mps::AbstractMPS,site::Int;epsilon=1E-13)
    l = mps.bdim[site-1]
    r = mps.bdim[site]
    A = reshape(mps[site],(l, r*mps.S))
    if USE_CUDA
        A = CuArray(A)
    end
    U, S, V = svd!(A)
    A = nothing
    Dnew = min(sum(S.>epsilon))
    V = copy(adjoint(V))[1:Dnew,:]
    V = copy(reshape(V,(Dnew,mps.S,:)))
    mps[site] = Array(V)
    temp = reshape(mps[site-1],mps.bdim[site-2]*mps.S,mps.bdim[site-1])
    if USE_CUDA
        temp = CuArray(temp)
    end
    mps[site-1] = Array(reshape(temp* U[:,1:Dnew] *Diagonal(S[1:Dnew]),(mps.bdim[site-2], mps.S, Dnew)))
    mps.bdim[site-1] = Dnew
    return S
end

function _overlap(mps1::AbstractMPS,mps2::AbstractMPS,::Type{CuArray})
    epsilon = 1E-13
    E = cu(transpose(reshape(mps1[1],(mps1.S, mps1.bdim[1])))) * cu(reshape(mps2[1],(mps2.S, mps2.bdim[1])))
    res = 0.0
    for i=2:mps1.L
        #E = torch.einsum('ab,ade,bdf->ef', (E, mps1[i], mps2[i])) # einsum version of following three lines
        a=size(mps1[i])[1]
        e=size(mps1[i])[3]
        E = transpose(reshape(transpose(E) * reshape(cu(mps1[i]),a,:),:,e))
        E = E * reshape(cu(mps2[i]),:,mps2.bdim[i])
        s = norm(E)
        if abs(s)< epsilon
            return s
        end
        res += log(s)
        E = E./s # devided by norm
    end
    temp = sum(E)
    @assert abs(imag(temp))<1E-10 # TODO: we do not calculate non-real overlap. May need complex support
    return temp*exp(res)
end

function _overlap(mps1::AbstractMPS,mps2::AbstractMPS,::Type{Array})
    epsilon = 1E-13
    E = transpose(reshape(mps1[1],(mps1.S, mps1.bdim[1]))) * reshape(mps2[1],(mps2.S, mps2.bdim[1]))
    res = 0.0
    for i=2:mps1.L
        #E = torch.einsum('ab,ade,bdf->ef', (E, mps1[i], mps2[i])) # einsum version of following three lines
        a=size(mps1[i])[1]
        e=size(mps1[i])[3]
        E = transpose(reshape(transpose(E) * reshape(mps1[i],a,:),:,e))
        E = E * reshape(mps2[i],:,mps2.bdim[i])
        s = norm(E)
        if abs(s)< epsilon
            return s
        end
        res += log(s)
        E = E./s # devided by norm
    end
    temp = sum(E)
    @assert abs(imag(temp))<1E-10 # TODO: we do not calculate non-real overlap. May need complex support
    return temp*exp(res)
end
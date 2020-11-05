# This file contain lowerest level-1 function
"""
    see multiply!
"""
function single_tensor_multiply(mpo_tensor::AbstractArray{T,4} where T<:Number,mps_tensor::AbstractArray{T,3} where T<:Number)
    (l,u,r,d)=size(mpo_tensor)
    (a,d,b)=size(mps_tensor)
    temp_site = reshape(mpo_tensor,:,d)*reshape(permutedims(mps_tensor,(2,1,3)),d,:) #lurab
    temp_site = permutedims(reshape(temp_site,(l,u,r,a,b)),(1,4,2,3,5))
    return temp_site
end

"""
    sweep in a single site
"""
function single_tensor_sweep!(mps::abstractMPS,site::Int,::LeftNormalization)
    l=mps.bdim[site-1] # left bond dimension
    r=mps.bdim[site]   # current bond dimension
    A=reshape(mps[site],(l*mps.S,r)) # A is a matrix unfolded from the current tensor
    U,R = qr!(A) # here we intent to do QR = A. However there is no BP, so we do SVD instead 
    s = norm(R)
    R = R./s # devided by norm
    res = log(s)
    U = copy(typeof(R)(U)) 
    Dnew = size(R)[1]

    U = reshape(U ,l,mps.S,Dnew)
    mps[site] = U   # U is LinearAlgebra.QRCompactWYQ type,it is not normal array
    R = R*reshape(mps[site+1],r,:)
    mps[site+1] = reshape(R,:,mps.S,mps.bdim[site+1])
    mps.bdim[site] = Dnew
    return res
end

function single_tensor_sweep!(mps::abstractMPS,site::Int,::RightNormalization)
    l = mps.bdim[site-1]
    r = mps.bdim[site]
    A = copy(reshape(mps[site],(l, r*mps.S)))
    A = copy(transpose(A))
    U,R = qr!(A) # UR = A^{T} R^{T}U^{T} = A
    Dnew = size(R)[1]
    U = transpose(copy(typeof(R)(U)))
    R = transpose(R)
    mps[site] = reshape(U,(Dnew,mps.S,r))
    mps[site-1] = reshape(reshape(mps[site-1],mps.bdim[site-2]*mps.S,mps.bdim[site-1])* R ,(mps.bdim[site-2], mps.S, Dnew))
    mps.bdim[site-1] = Dnew
end

"""
    cut-off a single tensor
"""
function single_tensor_cutoff!(mps::abstractMPS,site::Int,Dcut::Int,::LeftNormalization;epsilon=0)
    l = mps.bdim[site-1]
    r = mps.bdim[site]
    A = reshape(mps[site],(l, r*mps.S))
    U, S, V = svd!(A)
    Dnew = min(Dcut, sum(S.>epsilon))
    res = sum(S[Dnew+1:min(l,r*mps.S)])
    V = copy(adjoint(V))[1:Dnew,:]
    V = reshape(V,(Dnew,mps.S,:))
    mps[site] = V
    mps[site-1] = reshape(reshape(mps[site-1],mps.bdim[site-2]*mps.S,mps.bdim[site-1])* U[:,1:Dnew] *diagm(S[1:Dnew]),(mps.bdim[site-2], mps.S, Dnew))
    mps.bdim[site-1] = Dnew
    return res
end

"""
    find the sigular value spectrum for a single tensor
"""
function single_tensor_spectrum!(mps::abstractMPS,site::Int;epsilon=1E-13)
    l = mps.bdim[site-1]
    r = mps.bdim[site]
    A = reshape(mps[site],(l, r*mps.S))
    U, S, V = svd!(A)
    Dnew = min(sum(S.>epsilon))
    V = copy(adjoint(V))[1:Dnew,:]
    V = copy(reshape(V,(Dnew,mps.S,:)))
    mps[site] = V
    mps[site-1] = reshape(reshape(mps[site-1],mps.bdim[site-2]*mps.S,mps.bdim[site-1])* U[:,1:Dnew] *diagm(S[1:Dnew]),(mps.bdim[site-2], mps.S, Dnew))
    mps.bdim[site-1] = Dnew
    return S
end
abstract type Normalization end
struct RightNormalization <: Normalization end
struct LeftNormalization <: Normalization end
struct MixNormalization <: Normalization end

export normalization!,RightNormalization,LeftNormalization,MixNormalization,Normalization
"""
    normalization for MPS. This function will set the norm of a mps to 1.
"""
function normalization!(mps::abstractMPS,::LeftNormalization)
    for site =1:mps.L-1
        l=mps.bdim[site-1] # left bond dimension
        r=mps.bdim[site]   # current bond dimension
        A=reshape(mps[site],(l*mps.S,r)) # A is a matrix unfolded from the current tensor
        U,R = qr!(A) # here we intent to do QR = A. However there is no BP, so we do SVD instead 
        s = norm(R)
        R = R./s # devided by norm
        U = Array(U) 
        Dnew = size(R)[1]

        U = reshape(U ,l,mps.S,Dnew)
        mps[site] = U   # U is LinearAlgebra.QRCompactWYQ type,it is not normal array
        R = R*reshape(mps[site+1],r,:)
        mps[site+1] = reshape(R,:,mps.S,mps.bdim[site+1])
        mps.bdim[site] = Dnew
    end
    mps[mps.L] = mps[mps.L]/sqrt(sum(mps[mps.L].*mps[mps.L]))
end

#TODO: need to add other type of normalization
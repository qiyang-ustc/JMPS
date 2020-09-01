export entropy

function entropy(mps::MPS,epsilon::Float64=1E-13)
    # Calculate the von-Neumann entanglement entropy of an MPS
    res = 0.0
    entropy = 0.0
    #from left to right, svd 
    for site =1:mps.L-1
        l=mps.bdim[site-1] # left bond dimension
        r=mps.bdim[site]   # current bond dimension
        A=reshape(mps[site],(l*mps.S,r)) # A is a matrix unfolded from the current tensor
        U, R = qr(A) # here we intent to do QR = A. However there is no BP, so we do SVD instead 
        s = norm(R)
        res = res + log(s)
        R = R./s # devided by norm
        Dnew = size(R)[1]
        mps[site] = reshape(U[:,1:Dnew],l,mps.S,Dnew)   # U is LinearAlgebra.QRCompactWYQ type,it is not normal array
        mps[site+1] = reshape(R*reshape(mps[site+1],r,:),:,mps.S,mps.bdim[site+1])
        mps.bdim[site] = Dnew
    end
    for site = mps.L:-1:2 
        l = mps.bdim[site-1]
        r = mps.bdim[site]
        A = reshape(mps[site],(l, r*mps.S))
        U, S, V = svd(A)
        @assert (site==mps.L) | (abs(sum(S.*S)-1.0)<1E-13)  #check normalization 
        S = S./sqrt(sum(S.*S))  #normalize MPS
        entropy += - S.*log.(S)   #entanglement entropy between site with site-1
        Dnew = sum(S.>epsilon)
        # print (S[:Dnew])
        mps[site] = reshape(transpose(V[:, 1:Dnew]),(Dnew,mps.S,:))
        mps[site-1] = reshape(reshape(mps[site-1],mps.bdim[site-2]*mps.S,mps.bdim[site-1])* U[:,1:Dnew] *diagm(S[1:Dnew]),(mps.bdim[site-2], mps.S, Dnew))
        mps.bdim[site-1] = Dnew
    #print(mps.bdim)
    #print ('addition:', res)
    end
    return entropy
end

export entropy
"""
This function is not tested!!!
"""
function entropy(mps::MPS,bond_id::Int)
    # Calculate the von-Neumann entanglement entropy of an MPS
    #from left to right, svd 
    res = 0.0
    @assert bond_id>0 && bond_id <= mps.L
    for site =1:bond_id-1
        l=mps.bdim[site-1] # left bond dimension
        r=mps.bdim[site]   # current bond dimension
        A=reshape(mps[site],(l*mps.S,r)) # A is a matrix unfolded from the current tensor
        U, R = qr(A) # here we intent to do QR = A. However there is no BP, so we do SVD instead 
        U = Array(U)
        s = norm(R)
        res = res + log(s)
        R = R./s # devided by norm
        Dnew = size(R)[1]
        mps[site] = reshape(U[:,1:Dnew],l,mps.S,Dnew)   # U is LinearAlgebra.QRCompactWYQ type,it is not normal array
        mps[site+1] = reshape(R*reshape(mps[site+1],r,:),:,mps.S,mps.bdim[site+1])
        mps.bdim[site] = Dnew
    end
    for site = mps.L:-1:bond_id+2
        l = mps.bdim[site-1]
        r = mps.bdim[site]
        A = reshape(mps[site],(l, r*mps.S))
        A = transpose(A)
        U,R = qr(A) # UR = A^{T} R^{T}U^{T} = A
        Dnew = size(R)[1]
        U = transpose(Array(U))
        R = transpose(R)
        mps[site] = reshape(U,(Dnew,mps.S,r))
        mps[site-1] = reshape(reshape(mps[site-1],mps.bdim[site-2]*mps.S,mps.bdim[site-1])* R ,(mps.bdim[site-2], mps.S, Dnew))
        mps.bdim[site-1] = Dnew
        #print(mps.bdim)
        #print ('addition:', res)
    end

    l = reshape(mps[bond_id],mps.bdim[bond_id-1]*mps.S,mps.bdim[bond_id])
    r = reshape(mps[bond_id+1],mps.bdim[bond_id],mps.S*mps.bdim[bond_id+1])
    U,S,V = svd(l*r)

    S = S./sqrt(sum(S.*S))  #normalize MPS
    S .+ 1E-100
    @show S
    return sum(- S.*S .* log.(S.*S))
end

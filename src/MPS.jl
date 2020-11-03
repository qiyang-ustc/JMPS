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


function compress!(mps::abstractMPS,Dcut::Int,epsilon=1E-13)
    # cut a mps up to given bond dimension
    res = zeros(mps.L-1)
    #from left to right, svd 
    for site =1:mps.L-1
        l=mps.bdim[site-1] # left bond dimension
        r=mps.bdim[site]   # current bond dimension
        A=reshape(mps[site],(l*mps.S,r)) # A is a matrix unfolded from the current tensor
        U,R = qr!(A) # here we intent to do QR = A. However there is no BP, so we do SVD instead 
        s = norm(R)
        # @show s
        R = R./s # devided by norm

        U = Array(U) # convert strange type into array
        Dnew = size(R)[1]
        # The following line will cause critical overhead, type of U is strange,
        # (Never try to slice QRCompactWYQ)
        # U = @view U[:,1:Dnew] 
        
        U = reshape(U ,l,mps.S,Dnew)
        mps[site] = U   # U is LinearAlgebra.QRCompactWYQ type,it is not normal array
        R = R*reshape(mps[site+1],r,:)
        mps[site+1] = reshape(R,:,mps.S,mps.bdim[site+1])
        mps.bdim[site] = Dnew
    end
    #print (mps.bdim)
    #from right to left, svd
    for site = mps.L:-1:2 
        l = mps.bdim[site-1]
        r = mps.bdim[site]
        A = reshape(mps[site],(l, r*mps.S))
        U, S, V = svd!(A)
        Dnew = min(Dcut, sum(S.>epsilon))
        res[site-1] = sum(S[Dnew+1:min(l,r*mps.S)])
        V = @view adjoint(V)[1:Dnew,:]
        V = reshape(V,(Dnew,mps.S,:))
        mps[site] = V
        mps[site-1] = reshape(reshape(mps[site-1],mps.bdim[site-2]*mps.S,mps.bdim[site-1])* U[:,1:Dnew] *diagm(S[1:Dnew]),(mps.bdim[site-2], mps.S, Dnew))
        mps.bdim[site-1] = Dnew
    #print(mps.bdim)
    #print ('addition:', res)
    end
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


# You should not include or use this file directly
# it contains basic functionality of MPS.
export MPS,MPO,getindex,setindex!,overlap,multiply!,multiply,compress!,*,transpose,transpose!,dag,dag!,disp

abstract type TensorNetwork end
abstract type TensorArray <: TensorNetwork end
getindex(tensors::TensorArray,target::Int) = getindex(tensors.tensor,target)
setindex!(tensors::TensorArray,tensor::AbstractArray,target::Int) = setindex!(tensors.tensor,tensor,target)

#implement recycle index
# bidm[i] is bond dimension between tensors[i] and tensors[i+1]
struct Bdims
    L::Int
    bdim::Array{Int,1}
end
getindex(bdim::Bdims,target::Int) = getindex(bdim.bdim,(target+bdim.L-1) % bdim.L+1)
setindex!(bdim::Bdims,value::Int,target::Int) = setindex!(bdim.bdim,value,(target+bdim.L-1) % bdim.L + 1)

mutable struct MPS{FloatType<:Number} <: TensorArray
    L::Int #length
    S::Int #physical index
    bdim::Bdims
    tensor::Array{Array{FloatType,3},1}
end

function MPS(FloatType,L::Int,S::Int,D::Int)
    bdim = Bdims(L,append!([D for i=1:L-1],1))
    tensor = [zeros(FloatType,bdim[i-1],S,bdim[i]) for i=1:L]
    return MPS{FloatType}(L,S,bdim,tensor)
end

struct MPO{FloatType<:Number} <: TensorArray
    L::Int #length
    S::Int #physical index
    bdim::Bdims
    tensor::Array{Array{FloatType,4},1}
end

function MPO(FloatType,L::Int,S::Int,D::Int)
    bdim = Bdims(L,append!([D for i=1:L-1],1))
    tensor = [zeros(FloatType,bdim[i-1],S,bdim[i],S) for i=1:L]
    return MPO{FloatType}(L,S,bdim,tensor)
end

function transpose(mpo::MPO)
    temp_mpo = deepcopy(mpo)
    transpose!(temp_mpo)
    return temp_mpo
end

function transpose!(mpo::MPO)
    for i = 1:1:mpo.L
        mpo[i] = permutedims(mpo[i],[1,4,3,2])
    end
    return mpo
end

"""
overlap of two mps: mps1 and mps2.
the same as transpose(mps2)*mps1
"""
function overlap(mps1::MPS,mps2::MPS)
    epsilon = 1E-13
    E = transpose(reshape(mps1[1],(mps1.S, mps1.bdim[1]))) * reshape(mps2[1],(mps2.S, mps2.bdim[1]))
    scale = SciNum(1.0)
    for i=2:mps1.L
        #E = torch.einsum('ab,ade,bdf->ef', (E, mps1[i], mps2[i])) # einsum version of following three lines
        a=size(mps1[i])[1]
        e=size(mps1[i])[3]
        # print('E.t()=',E.t())
        # print('mps1[i]=',mps1[i].view(a,-1))
        # print(mps2[i].contiguous().view([-1,mps2[i].shape[2]]))
        E = transpose(reshape(transpose(E) * reshape(mps1[i],a,:),:,e))
        E = E * reshape(mps2[i],:,mps2.bdim[i])
        # print('E(end)=',E)
        #normalize E
        s = norm(E)
        if abs(s)< epsilon
            return SciNum(s)
        end
        scale = scale*SciNum(s)
        E = E./s # devided by norm
    end
    temp = sum(E)
    @assert abs(imag(temp))<1E-10 # we do not calculate non-real overlap
    return SciNum(real(temp))*scale
end

function multiply(mpo::MPO,mps::MPS)
    temp_mps = deepcopy(mps)
    multiply!(mpo,temp_mps)
    return temp_mps
end

"""
for an MPO => T, multiply(mpo,mps)= T|mps>
"""
multiply!(mpo::MPO,mps::MPS) = multiply!(mpo,mps,1,mps.L)

"""
Multiply only from site s to site e
Please check s,e in 1:mps.L
"""
function multiply!(mpo::MPO,mps::MPS,s::Int,e::Int) 
    for site = (s>e ? (e:s) : (s:e))
        # @show size(mpo[site]),size(mps[site])
	    temp_site = ein"lurd,adb->laurb"(mpo[site], mps[site]) # einsum version of following 4 lines
        # (l,u,r,d)=size(mpo[site])
        # (a,d,b)=size(mps[site])
        # temp_site = reshape(mpo[site],:,d)*reshape(permutedims(mps[site],(2,1,3)),d,:) #lurab
        # temp_site = permutedims(reshape(temp_site,(l,u,r,a,b)),(1,4,2,3,5))
        mps.bdim[site-1] = mpo.bdim[site-1]*mps.bdim[site-1] # only change left 
        mps[site] = reshape(temp_site,(mps.bdim[site-1], mps.S,:))
    end
    return mps
end

*(mps1::MPS,mps2::MPS) = overlap(dag(mps1),mps2)
*(mpo::MPO,mps::MPS) = multiply(mpo,mps)
*(mps::MPS,mpo::MPO) = multiply(transpose(mpo),mps)

function compress!(mps::MPS,Dcut::Int,epsilon=1E-13)
    # cut a mps up to given bond dimension
    res = 0.0
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
        res += sum(S[Dnew+1:min(l,r*mps.S)])
        V = @view V[:,1:Dnew]
        V = reshape(adjoint(V),(Dnew,mps.S,:))
        mps[site] = V
        mps[site-1] = reshape(reshape(mps[site-1],mps.bdim[site-2]*mps.S,mps.bdim[site-1])* U[:,1:Dnew] *diagm(S[1:Dnew]),(mps.bdim[site-2], mps.S, Dnew))
        mps.bdim[site-1] = Dnew
    #print(mps.bdim)
    #print ('addition:', res)
    end
    return res
end

function dag!(mps::MPS{T}) where T<:Complex
    for i=1:1:mps.L
        mps[i] = conj.(mps[i])
    end
    return mps
end

function dag(mps::MPS{T}) where T<:Complex
    temp = deepcopy(mps)
    dag!(temp)
    return temp
end

dag(mps::MPS{T}) where T<: Real = mps
dag!(mps::MPS{T}) where T<: Real = mps

function disp(mps::MPS,array::Array{Int,1})
    @assert mps.L == size(array)[1]
    t = transpose(mps[1][1,array[1],:])
    for i = 2:mps.L
        t = t*mps[i][:,array[i],:]
    end
    return t
end

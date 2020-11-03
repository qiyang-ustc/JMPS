#implement recycle index
# bidm[i] is bond dimension between tensors[i] and tensors[i+1]
struct Bdims
    L::Int
    bdim::Array{Int,1}
end
getindex(bdim::Bdims,target::Int) = getindex(bdim.bdim,(target+bdim.L-1) % bdim.L+1)
setindex!(bdim::Bdims,value::Int,target::Int) = setindex!(bdim.bdim,value,(target+bdim.L-1) % bdim.L + 1)
using CUDA
export randomMPS
function randomMPS(L::Int,S::Int,D::Int;FloatType=Float64,ArrayType=Array)
    mps = MPS(L,S,D;FloatType=Float64,ArrayType=Array)
    for i = 1:L
        mps[i] .= rand(size(mps[i])...)
    end
    if ArrayType==CuArray
        mps = cu(mps)
    end
    return mps
end

export randomMPO
function randomMPO(L::Int,S::Int,D::Int;FloatType=Float64,ArrayType=Array)
    mpo = MPO(L,S,D;FloatType=Float64,ArrayType=Array)
    for i = 1:L
        mpo[i] .= rand(size(mpo[i])...)
    end
    if ArrayType==CuArray
        mpo = cu(mpo)
    end
    return mpo
end
abstract type Normalization end
struct RightNormalization <: Normalization end
struct LeftNormalization <: Normalization end
struct MixNormalization <: Normalization end

export normalization!,RightNormalization,LeftNormalization,MixNormalization,Normalization
"""
    using QR sweep mps from left to right or from right to left
    low level API, should not be exposed to user.
    Sweep from left to right 
"""
function sweep!(mps::abstractMPS,endpoint::Int,::LeftNormalization)
    res = 0.0
    @assert endpoint<mps.L && endpoint > 1
    for site =1:endpoint
        res+=single_tensor_sweep!(mps,site,LeftNormalization())
    end
    return res
end

"""
    using QR sweep mps from left to right or from right to left
    low level API, should not be exposed to user.
    Sweep from right to left
"""
function sweep!(mps::abstractMPS,endpoint::Int,::RightNormalization)
    res = 0.0
    @assert endpoint<mps.L && endpoint > 1
    for site = mps.L:-1:endpoint
        res+=single_tensor_sweep!(mps,site,RightNormalization())
    end
    return res
end

"""
    using QR sweep mps from left to right or from right to left
    low level API, should not be exposed to user.
    Sweep from right to endpoint-1 and left to endpoint+1
"""
function sweep!(mps::abstractMPS,endpoint::Int,::MixNormalization)
    res = 0.0
    res += sweep!(mps,endpoint-1,LeftNormalization())
    res += sweep!(mps,endpoint+1,RightNormalization())
    return res
end

"""
    Cut-off subprogram for a Left normalization MPS.
"""
function cutoff!(mps::abstractMPS,Dcut::Int,::LeftNormalization)
    res = zeros(mps.L-1)
    for site = mps.L:-1:2 
        res[site-1] = single_tensor_cutoff!(mps,site,Dcut,LeftNormalization())
    end
    return res
end

"""
    normalization for MPS. This function will set the norm of a mps to 1.
        return res = log(s...)
"""
function normalization!(mps::abstractMPS,::LeftNormalization)
    res = sweep!(mps,mps.L-1,LeftNormalization())
    mps[mps.L] = mps[mps.L]/sqrt(sum(mps[mps.L].*mps[mps.L]))
    return res
end

function normalization!(mps::abstractMPS,::RightNormalization)
    res = sweep!(mps,2,RightNormalization())
    mps[1] = mps[1]/sqrt(sum(mps[1].*mps[1]))
    return res
end

function normalization!(mps::abstractMPS,site::Int,::MixNormalization)
    res = sweep!(mps,site,MixNormalization())
    mps[site] = mps[site]/sqrt(sum(mps[site].*mps[site]))
    return res
end
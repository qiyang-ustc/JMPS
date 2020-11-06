export entropy,spectrum
"""
    Calculate the von-Neumann entanglement entropy of an MPS at bond_id
"""
function entropy(mps::AbstractMPS,bond_id::Int)
    S = spectrum(mps,bond_id)
    S = S./sqrt(sum(S.*S))  #normalize MPS
    S = S .+ 1E-100
    return sum(- S.*S .* log.(S.*S))
end

"""
    Calculate von-Neumann entanglement entropy of every bonds
"""
function entropy(mps::AbstractMPS,epsilon=1E-13)
    function spectrum2entropy(S::AbstractVector)
        t = S./sqrt(sum(S.*S))  #normalize MPS
        t = t .+ 1E-100
        return sum(- t.*t .* log.(t.*t))
    end
    # Calculate the von-Neumann entanglement entropy of an MPS
    #from left to right, qr
    v_entropy = zeros(Float64,mps.L-1)
    sweep!(mps,mps.L-1,LeftNormalization())
    #print (mps.bdim)
    #from right to left, svd
    for site = mps.L:-1:2 
        S = single_tensor_spectrum!(mps,site)
        v_entropy[site-1] = spectrum2entropy(S)
    end
    return v_entropy
end

"""
Calculate Spectrum in MPS: from site-bond_id to site-bond_id+1
"""
function spectrum(mps::AbstractMPS,bond_id::Int)
    res = 0.0
    @assert bond_id>0 && bond_id <= mps.L
    normalization!(mps,bond_id)
    U,S,V = svd(reshape(mps[bond_id],mps.bdim[bond_id-1]*mps.S,mps.bdim[bond_id]))
    return S
end 
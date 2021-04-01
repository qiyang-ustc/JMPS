#debug utils
export check_bdim
function check_bdim(mps::MPS)
    for i = 1:mps.L
        if size(mps[i])!= (mps.bdim[i-1],mps.S,mps.bdim[i])
            print("Array Size Error: Tensor:$i, bdim size is $(mps.bdim[i-1]),$(mps.S),$(mps.bdim[i]), in fact it is $(size(mps[i]))\n")
        end
    end
    return nothing
end

export array
"""
store wave function as a tensor
    Notice: this is a quite costful function if N-> large!!!
"""
function array(mps::MPS;field=1:S)
    norm = Float64(mps*mps)
    ID = CartesianIndices(tuple([field for i=1:N]...))
    tensor = zeros(FloatType,[last(field) for i=1:N]...)
    for id in ID
        tensor[id] = disp(mps,id)
    end
    return tensor
end
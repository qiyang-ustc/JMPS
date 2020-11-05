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
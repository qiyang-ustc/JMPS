# You should not include or use this file directly
# it contains basic functionality of MPS.
export MPS,MPO,getindex,setindex!,overlap,multiply!,multiply,compress!,*,transpose,transpose!,dag,dag!,disp

abstract type TensorNetwork end
abstract type TensorArray <: TensorNetwork end
abstract type AbstractMPS <: TensorArray end 
abstract type abstractMPO <: TensorArray end
getindex(tensors::TensorArray,target::Int) = getindex(tensors.tensor,target)
setindex!(tensors::TensorArray,tensor::AbstractArray,target::Int) = setindex!(tensors.tensor,tensor,target)
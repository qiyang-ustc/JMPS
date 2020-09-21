module JMPS
    import Base.getindex,Base.setindex!,Base.*,Base.transpose
    using LinearAlgebra,TensorOperations,OMEinsum

    include("./SciNum.jl")
    using .SciNumModule
    include("./JMPS_BASIC.jl") # it contains basic functionality of MPS.
    include("./JMPS_PHY.jl")
end

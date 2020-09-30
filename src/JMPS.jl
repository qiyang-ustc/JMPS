module JMPS
    import Base.getindex,Base.setindex!,Base.*,Base.transpose
    using LinearAlgebra,TensorOperations,OMEinsum

    using Reexport
    include("./SciNum.jl")
    @reexport using JMPS.SciNumModule
    include("./JMPS_BASIC.jl") # it contains basic functionality of MPS.
    include("./JMPS_PHY.jl") # entropy
end
 
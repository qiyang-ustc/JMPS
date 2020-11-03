# Scientific Number for result output
module SciNumModule
    export /,SciNum,-,+,*
    export Float64,display
    mutable struct SciNum <: Real
        s::Float64 #Scale
        mp::Float64 #minus or plus
        SciNum(s::Float64,mp::Float64) = mp>0 ? new(s,1.0) : new(s,-1.0)
    end

    import Base.sign
    export sign,logscale
    sign(n::SciNum) = sign(n.mp)
    logscale(n::SciNum) = n.s

    import Base:+,-,*,/,abs
    abs(a::SciNum) = exp(a.s)
    /(a::SciNum,b::SciNum) = SciNum(a.s-b.s,a.mp*b.mp)
    *(a::SciNum,b::SciNum) = SciNum(a.s+b.s,a.mp*b.mp)
    +(a::SciNum,b::SciNum) = SciNum(Float64(a)+Float64(b))
    -(a::SciNum,b::SciNum) = SciNum(Float64(a)-Float64(b))

    import Base.display,Base.promote,Base.Float64,Base.promote_rule,Base.convert
    export promote

    isfinite(a::SciNum) = isfinite(a.s)
    convert(::Type{Float64},n::SciNum) = exp(n.s)*n.mp
    convert(::Type{SciNum},n::Float64) = abs(n)<1E-13 ?  SciNum(-Inf,1.0) : SciNum(log(abs(n)),sign(n))

    promote_rule(::Type{Float64},::Type{SciNum}) = Float64
    Float64(n::SciNum) = convert(Float64,n)
    SciNum(n::Float64) = convert(SciNum,n)
    display(n::SciNum) = n.mp>0 ? display("exp($(n.s))") : display("-exp($(n.s))")
end

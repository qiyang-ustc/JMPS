# Scientific Number for result output
export /,tofloat

mutable struct SciNum <: Real
    c::Float64 #coeffiecient
    s::Float64 #Scale
end

import Base./
/(a::SciNum,b::SciNum) = SciNum(a.c/b.c,a.s-b.s)

tofloat(n::SciNum) = n.c*exp(n.s)


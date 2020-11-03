using CUDA
using LinearAlgebra
import LinearAlgebra:svd,svd!
svd!(a::CuArray) = CUDA.CUSOLVER.svd(a)
# Array(a::CuArray) = CuArray(a)

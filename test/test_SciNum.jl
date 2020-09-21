a = SciNum(1.0,1.0)
b = SciNum(1.0,-1.0)
@testset "SciNum.jl" begin
    tol = 1E-13    

    @test abs(Float64(a)- exp(1.0))<tol
    @test abs(Float64(b)+ exp(1.0))<tol

    @test abs(exp(1.0)-Float64(a))<tol
    @test abs(exp(1.0)+Float64(b))<tol
    @test abs(Float64(a+b))<tol
    @test abs(Float64(a*b) +exp(2.0))<tol

    @test abs(a- exp(1.0))<tol
    @test abs(b+ exp(1.0))<tol
    @test abs(a+b)<tol
    @test (abs(a*b+exp(2.0)))<tol
    @test logscale(a)≈1.0
    @test logscale(b)≈1.0
    @test logscale(SciNum(1.0)/b)≈-1.0
end
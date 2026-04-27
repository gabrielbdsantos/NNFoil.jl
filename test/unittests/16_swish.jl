@testset "swish" begin
    @test NNFoil.swish(0.0) == 0.0
    @test NNFoil.swish(1.0, 2.0) ≈ 1 / (1 + exp(-2.0))

    @test NNFoil.swish(1.0f0) ≈ 1.0f0 / (1.0f0 + exp(-1.0f0))
    @test NNFoil.swish(1.0f0, 2.0f0) ≈ 1.0f0 / (1.0f0 + exp(-2.0f0))

    x = [-2.0, 0.0, 2.0]
    expected = x ./ (1 .+ exp.(-x))
    @test NNFoil.swish.(x) ≈ expected
end

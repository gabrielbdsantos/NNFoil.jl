@testset "bernstein" begin
    x = [0.0, 0.5, 1.0]
    @test NNFoil.bernstein(x, 0, 2) ≈ [1.0, 0.25, 0.0]
    @test NNFoil.bernstein(x, 1, 2) ≈ [0.0, 0.5, 0.0]
    @test NNFoil.bernstein(x, 2, 2) ≈ [0.0, 0.25, 1.0]
    @test NNFoil.bernstein(0.5, 1, 2) ≈ 0.5
end

@testset "sigmoid" begin
    @test NNFoil.sigmoid(0.0) == 0.5
    @test NNFoil.sigmoid(1.0) > 0.5
    @test NNFoil.sigmoid(-1.0) < 0.5
    @test 0.0 <= NNFoil.sigmoid(1.0e6) <= 1.0
    @test 0.0 <= NNFoil.sigmoid(-1.0e6) <= 1.0
    @test NNFoil.sigmoid(1.0e6) ≈ 1.0
    @test NNFoil.sigmoid(-1.0e6) < 1.0e-300
end

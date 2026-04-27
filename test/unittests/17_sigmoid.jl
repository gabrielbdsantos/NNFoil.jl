@testset "sigmoid" begin
    @test NNFoil.sigmoid(0.0) == 0.5
    @test NNFoil.sigmoid(1.0) > 0.5
    @test NNFoil.sigmoid(-1.0) < 0.5
    @test 0.0 <= NNFoil.sigmoid(1.0e6) <= 1.0
    @test 0.0 <= NNFoil.sigmoid(-1.0e6) <= 1.0
    @test NNFoil.sigmoid(1.0e6) ≈ 1.0
    @test NNFoil.sigmoid(-1.0e6) < 1.0e-300

    @test NNFoil.sigmoid(0.0f0) isa Float32
    @test NNFoil.sigmoid(0.0f0) == 0.5f0
    @test NNFoil.sigmoid(1.0f6) ≈ 1.0f0
    @test NNFoil.sigmoid(-1.0f6) < 1.0f-35
end

@testset "normalize_coordinates!" begin
    coords = [2.0 0.5; 4.0 1.0; 6.0 -1.0]
    result = NNFoil.normalize_coordinates!(coords)

    @test result === coords
    @test minimum(coords[:, 1]) == 0.0
    @test maximum(coords[:, 1]) == 1.0
    @test coords[1, 2] == 0.125
    @test coords[2, 2] == 0.25
    @test coords[3, 2] == -0.25
end

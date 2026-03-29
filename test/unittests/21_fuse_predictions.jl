@testset "fuse_predictions!" begin
    y = [1.0 3.0; 2.0 4.0]
    y_flipped = [3.0 1.0; 4.0 2.0]

    NNFoil.fuse_predictions!(y, y_flipped)
    @test y == [2.0 2.0; 3.0 3.0]
end

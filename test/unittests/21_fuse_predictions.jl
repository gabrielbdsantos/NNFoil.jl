@testset "fuse_predictions!" begin
    y = [1.0 3.0; 2.0 4.0]
    y_flipped = [3.0 1.0; 4.0 2.0]

    NNFoil.fuse_predictions!(y, y_flipped)
    @test y == [2.0 2.0; 3.0 3.0]

    @testset "zero allocation" begin
        y_alloc = [1.0 3.0; 2.0 4.0]
        y_flipped_alloc = [3.0 1.0; 4.0 2.0]
        fuse_alloc(yy, yy_flipped) = @allocated NNFoil.fuse_predictions!(yy, yy_flipped)

        NNFoil.fuse_predictions!(y_alloc, y_flipped_alloc)
        @test fuse_alloc(y_alloc, y_flipped_alloc) == 0
    end
end

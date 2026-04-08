@testset "flip_outputs!" begin
    y = [
        1.0 2.0
        3.0 4.0
        5.0 6.0
        7.0 8.0
        9.0 10.0
        11.0 12.0
    ]
    tmp = zeros(2)

    NNFoil.flip_outputs!(y, tmp)

    @test y[2, :] == [-3.0, -4.0]
    @test y[4, :] == [-7.0, -8.0]
    @test y[5, :] == [11.0, 12.0]
    @test y[6, :] == [9.0, 10.0]

    @testset "zero allocation" begin
        y_alloc = reshape(collect(1.0:12.0), 6, 2)
        tmp_alloc = zeros(2)
        flip_alloc(yy, tmp) = @allocated NNFoil.flip_outputs!(yy, tmp)

        NNFoil.flip_outputs!(y_alloc, tmp_alloc)
        @test flip_alloc(y_alloc, tmp_alloc) == 0
    end
end

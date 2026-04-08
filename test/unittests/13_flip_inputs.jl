@testset "flip_inputs!" begin
    x = reshape(collect(1.0:50.0), 25, 2)
    x_original = copy(x)
    NNFoil.flip_inputs!(x)

    @inbounds for i in axes(x, 2)
        for j in 1:8
            @test x[j, i] == -x_original[8 + j, i]
            @test x[8 + j, i] == -x_original[j, i]
        end
        @test x[17, i] == -x_original[17, i]
        @test x[18, i] == x_original[18, i]
        @test x[19, i] == -x_original[19, i]
        @test x[24, i] == x_original[25, i]
        @test x[25, i] == x_original[24, i]
    end

    NNFoil.flip_inputs!(x)
    @test x == x_original

    @test_throws DimensionMismatch NNFoil.flip_inputs!(zeros(24, 1))

    @testset "zero allocation" begin
        x_alloc = reshape(collect(1.0:50.0), 25, 2)
        flip_alloc(xx) = @allocated NNFoil.flip_inputs!(xx)

        NNFoil.flip_inputs!(x_alloc)
        @test flip_alloc(x_alloc) == 0
    end
end

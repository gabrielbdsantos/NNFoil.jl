@testset "flip_inputs!" begin
    @testset "batched input path" begin
        x_matrix = reshape(collect(1.0:50.0), 25, 2)
        x_original = copy(x_matrix)
        NNFoil.flip_inputs!(x_matrix)

        @inbounds for i in axes(x_matrix, 2)
            for j in 1:8
                @test x_matrix[j, i] == -x_original[8 + j, i]
                @test x_matrix[8 + j, i] == -x_original[j, i]
            end
            @test x_matrix[17, i] == -x_original[17, i]
            @test x_matrix[18, i] == x_original[18, i]
            @test x_matrix[19, i] == -x_original[19, i]
            @test x_matrix[24, i] == x_original[25, i]
            @test x_matrix[25, i] == x_original[24, i]
        end

        NNFoil.flip_inputs!(x_matrix)
        @test x_matrix == x_original

        @test_throws DimensionMismatch NNFoil.flip_inputs!(zeros(24, 1))
    end

    @testset "scalar input path" begin
        x_vec = collect(1.0:25.0)
        x_original = copy(x_vec)

        NNFoil.flip_inputs!(x_vec)

        for j in 1:8
            @test x_vec[j] == -x_original[8 + j]
            @test x_vec[8 + j] == -x_original[j]
        end
        @test x_vec[17] == -x_original[17]
        @test x_vec[18] == x_original[18]
        @test x_vec[19] == -x_original[19]
        @test x_vec[24] == x_original[25]
        @test x_vec[25] == x_original[24]

        NNFoil.flip_inputs!(x_vec)
        @test x_vec == x_original

        @test_throws DimensionMismatch NNFoil.flip_inputs!(zeros(24))
    end


    @testset "zero allocation" begin
        @testset "batched input" begin
            x_alloc = reshape(collect(1.0:50.0), 25, 2)
            flip_alloc(xx) = @allocated NNFoil.flip_inputs!(xx)

            NNFoil.flip_inputs!(x_alloc)
            @test flip_alloc(x_alloc) == 0
        end

        @testset "scalar input" begin
            x_alloc = collect(1.0:25.0)
            flip_alloc(xx) = @allocated NNFoil.flip_inputs!(xx)

            NNFoil.flip_inputs!(x_alloc)
            @test flip_alloc(x_alloc) == 0
        end
    end
end

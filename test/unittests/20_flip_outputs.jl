@testset "flip_outputs!" begin
    @testset "batched input path" begin
        y = [
            1.0 2.0
            3.0 4.0
            5.0 6.0
            7.0 8.0
            9.0 10.0
            11.0 12.0
        ]
        NNFoil.flip_outputs!(y)

        @test y[2, :] == [-3.0, -4.0]
        @test y[4, :] == [-7.0, -8.0]
        @test y[5, :] == [11.0, 12.0]
        @test y[6, :] == [9.0, 10.0]
    end

    @testset "scalar input path" begin
        y_vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        NNFoil.flip_outputs!(y_vec)
        @test y_vec == [1.0, -2.0, 3.0, -4.0, 6.0, 5.0]
    end

    @testset "zero allocation" begin
        @testset "batched input" begin
            y_alloc = reshape(collect(1.0:12.0), 6, 2)
            flip_alloc(yy) = @allocated NNFoil.flip_outputs!(yy)

            NNFoil.flip_outputs!(y_alloc)
            @test flip_alloc(y_alloc) == 0
        end

        @testset "scalar input" begin
            y_alloc = collect(1.0:6.0)
            flip_alloc(yy) = @allocated NNFoil.flip_outputs!(yy)

            NNFoil.flip_outputs!(y_alloc)
            @test flip_alloc(y_alloc) == 0
        end
    end
end

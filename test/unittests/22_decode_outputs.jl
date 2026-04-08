@testset "decode_outputs!" begin
    y = [
        0.0 2.0
        4.0 -4.0
        2.0 3.0
        20.0 -20.0
        -1.0 2.0
        0.2 1.2
    ]

    NNFoil.decode_outputs!(y)

    @test y[1, 1] == 0.5
    @test y[2, :] == [2.0, -2.0]
    @test y[3, :] ≈ [1.0, exp(2.0)]
    @test y[4, :] == [1.0, -1.0]
    @test y[5, :] == [0.0, 1.0]
    @test y[6, :] == [0.2, 1.0]

    @testset "zero allocation" begin
        y_alloc = [
            0.0 2.0
            4.0 -4.0
            2.0 3.0
            20.0 -20.0
            -1.0 2.0
            0.2 1.2
        ]
        decode_alloc(yy) = @allocated NNFoil.decode_outputs!(yy)

        NNFoil.decode_outputs!(y_alloc)
        @test decode_alloc(y_alloc) == 0
    end
end

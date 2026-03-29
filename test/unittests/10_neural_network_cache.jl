@testset "NeuralNetworkCache" begin
    x = NNFoil.build_features(UNIT_KULFAN, UNIT_ALPHA, UNIT_REYNOLDS_SCALAR)

    cache = NNFoil.NeuralNetworkCache(UNIT_NETWORK_PARAMETERS, copy(x))
    @test size(cache.x) == size(x)
    @test size(cache.y, 2) == size(x, 2)
    @test cache.tmp_x[1] === cache.x
    @test cache.tmp_x_flipped[1] === cache.x_flipped
    @test length(cache.outputs.CL) == size(x, 2)

    x_flipped_expected = copy(x)
    NNFoil.flip_inputs!(x_flipped_expected)
    @test cache.x_flipped == x_flipped_expected

    cache2 = NNFoil.NeuralNetworkCache(
        UNIT_NETWORK_PARAMETERS,
        UNIT_KULFAN,
        UNIT_ALPHA,
        UNIT_REYNOLDS_SCALAR,
    )
    @test size(cache2.x) == (25, length(UNIT_ALPHA))
end

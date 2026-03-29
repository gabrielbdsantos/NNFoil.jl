@testset "confidence_correction!" begin
    x = NNFoil.build_features(UNIT_KULFAN, UNIT_ALPHA[1:6], UNIT_REYNOLDS_SCALAR)
    y_ref = NNFoil.forward(UNIT_NETWORK_PARAMETERS, x)
    y_cache = copy(y_ref)

    NNFoil.confidence_correction!(y_ref, x, UNIT_NETWORK_PARAMETERS)

    cache = NNFoil.NeuralNetworkCache(UNIT_NETWORK_PARAMETERS, copy(x))
    NNFoil.confidence_correction!(y_cache, x, cache)

    @test y_cache ≈ y_ref
end

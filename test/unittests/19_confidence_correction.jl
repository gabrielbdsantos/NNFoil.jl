@testset "confidence_correction!" begin
    x = NNFoil.build_features(UNIT_KULFAN, UNIT_ALPHA[1:6], UNIT_REYNOLDS_SCALAR)

    @testset "batched input features" begin
        y_ref = NNFoil.forward(UNIT_NETWORK_PARAMETERS, x)
        y_cache = copy(y_ref)

        NNFoil.confidence_correction!(y_ref, x, UNIT_NETWORK_PARAMETERS)

        cache = NNFoil.NeuralNetworkCache(UNIT_NETWORK_PARAMETERS, copy(x))
        NNFoil.confidence_correction!(y_cache, x, cache)

        @test y_cache ≈ y_ref
    end

    @testset "scalar input path" begin
        x_scalar = NNFoil.build_features(UNIT_KULFAN, 2.0, UNIT_REYNOLDS_SCALAR)
        y_scalar = NNFoil.forward(UNIT_NETWORK_PARAMETERS, x_scalar)
        y_matrix = NNFoil.forward(UNIT_NETWORK_PARAMETERS, reshape(x_scalar, :, 1))

        NNFoil.confidence_correction!(y_scalar, x_scalar, UNIT_NETWORK_PARAMETERS)
        NNFoil.confidence_correction!(y_matrix, reshape(x_scalar, :, 1), UNIT_NETWORK_PARAMETERS)

        @test y_scalar[1] ≈ y_matrix[1, 1]
    end

    @testset "zero allocation (cache path)" begin
        y_alloc = NNFoil.forward(UNIT_NETWORK_PARAMETERS, x)
        cache_alloc = NNFoil.NeuralNetworkCache(UNIT_NETWORK_PARAMETERS, copy(x))
        correction_alloc(y, xx, c) = @allocated NNFoil.confidence_correction!(y, xx, c)

        NNFoil.confidence_correction!(y_alloc, x, cache_alloc)
        @test correction_alloc(y_alloc, x, cache_alloc) == 0
    end
end

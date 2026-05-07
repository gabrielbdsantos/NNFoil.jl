@testset "NeuralNetworkCache" begin
    @testset "size constructor" begin
        cache_scalar = NNFoil.NeuralNetworkCache(UNIT_NETWORK_PARAMETERS, 1, Float32)
        @test cache_scalar.network_parameters === UNIT_NETWORK_PARAMETERS
        @test cache_scalar.x isa Vector{Float32}
        @test size(cache_scalar.x) == (25,)
        @test size(cache_scalar.y) == (6, 1)
        @test cache_scalar.x_both === nothing
        @test cache_scalar.y_both === nothing
        @test length(cache_scalar.outputs.CL) == 1

        cache_batch = NNFoil.NeuralNetworkCache(UNIT_NETWORK_PARAMETERS, 3, Float32)
        @test cache_batch.network_parameters === UNIT_NETWORK_PARAMETERS
        @test cache_batch.x isa AbstractMatrix{Float32}
        @test size(cache_batch.x) == (25, 3)
        @test size(cache_batch.y) == (6, 3)
        @test size(cache_batch.x_both) == (25, 6)
        @test size(cache_batch.y_both) == (6, 6)
        @test parent(cache_batch.x) === cache_batch.x_both
        @test parent(cache_batch.x_flipped) === cache_batch.x_both
        @test parent(cache_batch.y) === cache_batch.y_both
        @test parent(cache_batch.y_flipped) === cache_batch.y_both
        @test length(cache_batch.outputs.CL) == 3

        cache_scalar_f64 = NNFoil.NeuralNetworkCache(UNIT_NETWORK_PARAMETERS, 1)
        @test cache_scalar_f64.x isa Vector{Float64}

        cache_batch_f64 = NNFoil.NeuralNetworkCache(UNIT_NETWORK_PARAMETERS, 3)
        @show typeof(cache_batch_f64.x)
        @test cache_batch_f64.x isa AbstractMatrix{Float64}

        @test_throws ArgumentError NNFoil.NeuralNetworkCache(UNIT_NETWORK_PARAMETERS, 0)
        @test_throws ArgumentError NNFoil.NeuralNetworkCache(UNIT_NETWORK_PARAMETERS, -5)
    end

    @testset "input shape errors" begin
        matrix_error = try
            NNFoil.NeuralNetworkCache(UNIT_NETWORK_PARAMETERS, zeros(24, 2))
        catch err
            err
        end
        @test matrix_error isa DimensionMismatch
        @test matrix_error.msg ==
            "`x0` must be of size (25, *). An array of size (24, 2) was given."

        vector_error = try
            NNFoil.NeuralNetworkCache(UNIT_NETWORK_PARAMETERS, zeros(24))
        catch err
            err
        end
        @test vector_error isa DimensionMismatch
        @test vector_error.msg ==
            "`x` must have length 25. A vector of length 24 was given."
    end

    x = NNFoil.build_features(UNIT_KULFAN, UNIT_ALPHA, UNIT_REYNOLDS_SCALAR)

    cache = NNFoil.NeuralNetworkCache(UNIT_NETWORK_PARAMETERS, copy(x))
    @test size(cache.x) == size(x)
    @test size(cache.y, 2) == size(x, 2)
    @test cache.tmp_x[1] === cache.x
    @test cache.tmp_x_flipped[1] === cache.x_flipped
    @test length(cache.outputs.CL) == size(x, 2)

    if !(cache.x_both === nothing)
        C = size(cache.x, 2)

        @test size(cache.x_both, 2) == 2C
        @test (@view cache.x_both[:, 1:C]) == cache.x
        @test (@view cache.x_both[:, (C + 1):(2C)]) == cache.x_flipped

        @test cache.tmp_x_both[1] === cache.x_both
        @test isequal(@view(cache.y_both[:, 1:C]), cache.y)
        @test isequal(@view(cache.y_both[:, (C + 1):(2C)]), cache.y_flipped)
        @test parent(cache.y) === cache.y_both
        @test parent(cache.y_flipped) === cache.y_both
    end

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

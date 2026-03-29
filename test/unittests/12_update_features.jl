@testset "update_features!" begin
    @testset "cache update from Kulfan inputs" begin
        n_crit = 10.0
        xtr_upper = 0.3
        xtr_lower = 0.8

        cache = NNFoil.NeuralNetworkCache(
            UNIT_NETWORK_PARAMETERS,
            UNIT_KULFAN,
            UNIT_ALPHA,
            UNIT_REYNOLDS_SCALAR,
        )
        x_ref = cache.x
        x_flipped_ref = cache.x_flipped

        NNFoil.update_features!(
            cache,
            UNIT_KULFAN,
            UNIT_ALPHA,
            UNIT_REYNOLDS_VECTOR;
            n_crit,
            xtr_upper,
            xtr_lower,
        )

        @test cache.x === x_ref
        @test cache.x_flipped === x_flipped_ref
        @test cache.tmp_x[1] === cache.x
        @test cache.tmp_x_flipped[1] === cache.x_flipped

        x_expected = NNFoil.build_features(
            UNIT_KULFAN,
            UNIT_ALPHA,
            UNIT_REYNOLDS_VECTOR;
            n_crit,
            xtr_upper,
            xtr_lower,
        )
        @test cache.x == x_expected

        NNFoil.update_features!(
            cache;
            kulfan_parameters = UNIT_KULFAN,
            alpha = UNIT_ALPHA,
            Reynolds = UNIT_REYNOLDS_VECTOR,
            n_crit,
            xtr_upper,
            xtr_lower,
        )
        @test cache.x == x_expected

        x_flipped_expected = copy(cache.x)
        NNFoil.flip_inputs!(x_flipped_expected)
        @test cache.x_flipped == x_flipped_expected
    end

    @testset "cache update from feature matrix" begin
        cache = NNFoil.NeuralNetworkCache(
            UNIT_NETWORK_PARAMETERS,
            UNIT_KULFAN,
            UNIT_ALPHA,
            UNIT_REYNOLDS_SCALAR,
        )

        x_new = NNFoil.build_features(
            UNIT_KULFAN,
            UNIT_ALPHA,
            UNIT_REYNOLDS_VECTOR;
            n_crit = 8.5,
            xtr_upper = 0.4,
            xtr_lower = 0.6,
        )
        NNFoil.update_features!(cache, x_new)
        @test cache.x == x_new

        x_flipped_expected = copy(x_new)
        NNFoil.flip_inputs!(x_flipped_expected)
        @test cache.x_flipped == x_flipped_expected

        @test_throws DimensionMismatch NNFoil.update_features!(cache, x_new[:, 1:3])
    end

    @testset "validation checks" begin
        cache = NNFoil.NeuralNetworkCache(
            UNIT_NETWORK_PARAMETERS,
            UNIT_KULFAN,
            UNIT_ALPHA,
            UNIT_REYNOLDS_SCALAR,
        )

        @test_throws DimensionMismatch NNFoil.update_features!(
            cache,
            UNIT_KULFAN,
            1.0,
            5.0e6,
        )

        @test_throws DimensionMismatch NNFoil.update_features!(
            cache,
            UNIT_KULFAN,
            UNIT_ALPHA[1:3],
            UNIT_REYNOLDS_SCALAR,
        )

        @test_throws DimensionMismatch NNFoil.update_features!(
            cache,
            UNIT_KULFAN,
            1.0,
            UNIT_REYNOLDS_VECTOR[1:3],
        )

        @test_throws DimensionMismatch NNFoil.update_features!(
            cache,
            UNIT_KULFAN,
            UNIT_ALPHA,
            UNIT_REYNOLDS_VECTOR[1:3],
        )

        bad_upper = NNFoil.KulfanParameters(
            upper_weights = ones(7),
            lower_weights = ones(8),
            leading_edge_weight = UNIT_KULFAN.leading_edge_weight,
            trailing_edge_thickness = UNIT_KULFAN.trailing_edge_thickness,
        )

        @test_throws DimensionMismatch NNFoil.update_features!(
            cache,
            bad_upper,
            UNIT_ALPHA,
            UNIT_REYNOLDS_SCALAR,
        )

        @test_throws MethodError NNFoil.update_features!(
            unit_feature_matrix(3);
            kulfan_parameters = UNIT_KULFAN,
            alpha = UNIT_ALPHA[1:3],
            Reynolds = UNIT_REYNOLDS_SCALAR,
        )
    end

    @testset "zero allocation" begin
        cache = NNFoil.NeuralNetworkCache(
            UNIT_NETWORK_PARAMETERS,
            UNIT_KULFAN,
            UNIT_ALPHA,
            UNIT_REYNOLDS_SCALAR,
        )
        x = NNFoil.build_features(UNIT_KULFAN, UNIT_ALPHA, UNIT_REYNOLDS_SCALAR)

        cache_alloc(c, k, a, r) = @allocated NNFoil.update_features!(c, k, a, r)
        raw_alloc(c, xx) = @allocated NNFoil.update_features!(c, xx)

        NNFoil.update_features!(cache, UNIT_KULFAN, UNIT_ALPHA, UNIT_REYNOLDS_SCALAR)
        NNFoil.update_features!(cache, x)

        @test cache_alloc(cache, UNIT_KULFAN, UNIT_ALPHA, UNIT_REYNOLDS_SCALAR) == 0
        @test raw_alloc(cache, x) == 0
    end
end

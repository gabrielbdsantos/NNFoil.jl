@testset "evaluate and evaluate!" begin
    alpha = UNIT_ALPHA[1:7]
    Reynolds = UNIT_REYNOLDS_VECTOR[1:7]

    x = NNFoil.build_features(UNIT_KULFAN, alpha, Reynolds; n_crit = 8.5)
    out = NNFoil.evaluate(UNIT_NETWORK_PARAMETERS, x)

    cache = NNFoil.NeuralNetworkCache(UNIT_NETWORK_PARAMETERS, copy(x))
    NNFoil.evaluate!(cache)
    @test isapprox(cache.outputs, out; atol = 1.0e-12)

    out_from_kulfan = NNFoil.evaluate(
        UNIT_NETWORK_PARAMETERS,
        UNIT_KULFAN,
        alpha,
        Reynolds;
        n_crit = 8.5,
    )
    @test isapprox(out_from_kulfan, out; atol = 1.0e-12)

    @test all(0 .<= out.analysis_confidence .<= 1)
    @test all(0 .<= out.Top_Xtr .<= 1)
    @test all(0 .<= out.Bot_Xtr .<= 1)

    @testset "supported output channels" begin
        @test size(cache.y, 1) == NNFoil.SUPPORTED_OUTPUT_CHANNELS
        @test size(cache.y_flipped, 1) == NNFoil.SUPPORTED_OUTPUT_CHANNELS
    end

    @testset "Float32 consistency" begin
        params32 = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float32)
        kulfan32 = NNFoil.KulfanParameters(
            upper_weights = Float32.(UNIT_KULFAN.upper_weights),
            lower_weights = Float32.(UNIT_KULFAN.lower_weights),
            leading_edge_weight = Float32(UNIT_KULFAN.leading_edge_weight),
            trailing_edge_thickness = Float32(UNIT_KULFAN.trailing_edge_thickness),
        )
        alpha32 = Float32.(UNIT_ALPHA[1:7])
        re32 = Float32.(UNIT_REYNOLDS_VECTOR[1:7])
        x32 = NNFoil.build_features(kulfan32, alpha32, re32; n_crit = Float32(8.5))
        out32 = NNFoil.evaluate(params32, x32)
        cache32 = NNFoil.NeuralNetworkCache(params32, copy(x32))
        NNFoil.evaluate!(cache32)

        @test isapprox(cache32.outputs, out32; atol = 1.0f-5)
        @test eltype(cache32.y) == Float32
        @test eltype(cache32.tmp_x[2]) == Float32
    end

    @testset "zero allocation" begin
        cache = NNFoil.NeuralNetworkCache(
            UNIT_NETWORK_PARAMETERS,
            UNIT_KULFAN,
            UNIT_ALPHA,
            UNIT_REYNOLDS_SCALAR,
        )

        eval_alloc(c) = @allocated NNFoil.evaluate!(c)

        NNFoil.evaluate!(cache)
        @test eval_alloc(cache) == 0
    end
end

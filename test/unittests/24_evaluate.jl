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
end

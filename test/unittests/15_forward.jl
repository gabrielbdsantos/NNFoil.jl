@testset "forward and forward!" begin
    x = NNFoil.build_features(UNIT_KULFAN, UNIT_ALPHA[1:5], UNIT_REYNOLDS_SCALAR)
    y_ref = NNFoil.forward(UNIT_NETWORK_PARAMETERS, x)

    y_cache, tmp = NNFoil.allocate_forward_cache(UNIT_NETWORK_PARAMETERS, copy(x))
    NNFoil.forward!(y_cache, UNIT_NETWORK_PARAMETERS, tmp)

    @test y_cache ≈ y_ref

    x_scalar = NNFoil.build_features(UNIT_KULFAN, 3.0, 5.0e6)
    y_scalar = NNFoil.forward(UNIT_NETWORK_PARAMETERS, x_scalar)
    y_scalar_matrix = NNFoil.forward(UNIT_NETWORK_PARAMETERS, reshape(x_scalar, :, 1))

    @test y_scalar ≈ vec(y_scalar_matrix)
end

@testset "squared_mahalanobis_distance" begin
    x = NNFoil.build_features(UNIT_KULFAN, UNIT_ALPHA[1:4], UNIT_REYNOLDS_SCALAR)
    y_out = NNFoil.squared_mahalanobis_distance(UNIT_NETWORK_PARAMETERS, x)

    y_in = zeros(size(x, 2), 1)
    tmp1 = similar(x)
    tmp2 = similar(x)
    NNFoil.squared_mahalanobis_distance!(y_in, UNIT_NETWORK_PARAMETERS, x, tmp1, tmp2)

    @test y_in[:, 1] ≈ vec(y_out)
    @test all(y_in[:, 1] .>= 0)
end

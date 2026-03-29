@testset "build_features" begin
    x = NNFoil.build_features(
        UNIT_SYNTH_KULFAN,
        10.0,
        2.0e6;
        n_crit = 8.0,
        xtr_upper = 0.2,
        xtr_lower = 0.7,
    )

    @test length(x) == 25
    @test x[1:8] == UNIT_SYNTH_KULFAN.upper_weights
    @test x[9:16] == UNIT_SYNTH_KULFAN.lower_weights
    @test x[17] == UNIT_SYNTH_KULFAN.leading_edge_weight
    @test x[18] == UNIT_SYNTH_KULFAN.trailing_edge_thickness * 50
    @test x[19] ≈ sind(20.0)
    @test x[20] ≈ cosd(10.0)
    @test x[21] ≈ (1 - cosd(10.0)^2)
    @test x[22] ≈ (log(2.0e6) - 12.5) / 3.5
    @test x[23] ≈ (8.0 - 9) / 4.5
    @test x[24] == 0.2
    @test x[25] == 0.7

    alpha = [0.0, 10.0]
    x_matrix = NNFoil.build_features(UNIT_SYNTH_KULFAN, alpha, 3.0e6)
    @test size(x_matrix) == (25, 2)
    @test x_matrix[19, 2] ≈ sind(20.0)

    @test_throws DimensionMismatch NNFoil.build_features(
        UNIT_SYNTH_KULFAN,
        [0.0, 1.0],
        [3.0e6],
    )
end

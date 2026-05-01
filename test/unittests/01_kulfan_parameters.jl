@testset "KulfanParameters" begin
    manual = NNFoil.KulfanParameters(
        upper_weights = ones(8),
        lower_weights = -ones(8),
        leading_edge_weight = 0.02,
        trailing_edge_thickness = 0.01,
    )
    @test length(manual.upper_weights) == 8
    @test length(manual.lower_weights) == 8
    @test manual.leading_edge_weight == 0.02
    @test manual.trailing_edge_thickness == 0.01

    fitted = NNFoil.KulfanParameters(copy(UNIT_COORDINATES))
    @test length(fitted.upper_weights) == 8
    @test length(fitted.lower_weights) == 8
    @test fitted.trailing_edge_thickness >= 0

    @testset "negative TE fit fallback" begin
        x_upper = collect(range(1.0, 0.0; length = 40))
        x_lower = collect(range(0.0, 1.0; length = 40))[2:end]
        x = [x_upper; x_lower]

        params_with_negative_te = [
            collect(0.15:0.05:0.5);
            collect(-0.10:-0.05:-0.45);
            0.02;
            -0.08;
        ]

        y = NNFoil.cst_y_coordinates(x, params_with_negative_te, length(x_upper))
        fitted_negative_te = NNFoil.KulfanParameters(hcat(x, y))

        @test fitted_negative_te.trailing_edge_thickness == 0
        @test all(isfinite, fitted_negative_te.upper_weights)
        @test all(isfinite, fitted_negative_te.lower_weights)
        @test isfinite(fitted_negative_te.leading_edge_weight)
    end

    all_params = [
        fitted.upper_weights;
        fitted.lower_weights;
        fitted.leading_edge_weight;
        fitted.trailing_edge_thickness;
    ]
    @test all(isfinite, all_params)

    @test_throws DimensionMismatch NNFoil.KulfanParameters(
        upper_weights = ones(7),
        lower_weights = ones(8),
        leading_edge_weight = UNIT_KULFAN.leading_edge_weight,
        trailing_edge_thickness = UNIT_KULFAN.trailing_edge_thickness,
    )

    @test_throws DimensionMismatch NNFoil.KulfanParameters(
        upper_weights = ones(8),
        lower_weights = ones(7),
        leading_edge_weight = UNIT_KULFAN.leading_edge_weight,
        trailing_edge_thickness = UNIT_KULFAN.trailing_edge_thickness,
    )
end

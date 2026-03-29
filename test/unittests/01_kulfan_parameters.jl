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

    all_params = [
        fitted.upper_weights;
        fitted.lower_weights;
        fitted.leading_edge_weight;
        fitted.trailing_edge_thickness;
    ]
    @test all(isfinite, all_params)
end

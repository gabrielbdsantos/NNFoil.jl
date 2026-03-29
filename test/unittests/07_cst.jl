@testset "cst" begin
    x = [0.0, 0.3, 1.0]
    coefficients = [0.2, 0.1, 0.05]
    leading_edge_weight = 0.03
    trailing_edge_thickness = 0.02

    expected =
        NNFoil.class_function(x) .* NNFoil.shape_function(x, coefficients) .+
        x .* trailing_edge_thickness .+
        leading_edge_weight .* x .* max.(1 .- x, 0) .^ (length(coefficients) + 0.5)

    y = NNFoil.cst(x, coefficients, leading_edge_weight, trailing_edge_thickness)
    @test y ≈ expected
    @test y[1] == 0.0
    @test y[end] ≈ trailing_edge_thickness
end

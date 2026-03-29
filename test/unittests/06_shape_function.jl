@testset "shape_function" begin
    x = [0.0, 0.5, 1.0]
    coefficients = [1.0, 2.0, 3.0]

    expected =
        coefficients[1] .* (1 .- x) .^ 2 .+
        coefficients[2] .* (2 .* x .* (1 .- x)) .+
        coefficients[3] .* x .^ 2

    @test NNFoil.shape_function(x, coefficients) ≈ expected
    @test NNFoil.shape_function(x, zeros(3)) == zeros(3)
end

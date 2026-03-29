@testset "cst_y_coordinates" begin
    x = [1.0, 0.5, 0.0, 0.5, 1.0]
    parameters = [0.1, 0.2, -0.1, -0.2, 0.03, 0.02]

    y = NNFoil.cst_y_coordinates(x, parameters, 3)

    @test length(y) == length(x)
    @test y[1] ≈ 0.01
    @test y[end] ≈ -0.01
    @test y[2] > y[4]
end

@testset "class_function" begin
    x = [0.0, 0.25, 1.0]
    y = NNFoil.class_function(x)
    @test y ≈ sqrt.(x) .* (1 .- x)
    @test y[1] == 0.0
    @test y[end] == 0.0
    @test all(y .>= 0)
end

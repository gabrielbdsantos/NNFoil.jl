@testset "NeuralNetworkParameters" begin
    p64 = NNFoil.NeuralNetworkParameters(; model_size = :xsmall, T = Float64)
    @test eltype(p64.mean_inputs_scaled) == Float64
    @test size(p64.cov_inputs_scaled) == (25, 25)
    @test size(p64.inv_cov_inputs_scaled) == (25, 25)
    @test length(p64.weights) > 0
    @test length(p64.weights) == length(p64.biases)
    @test size(p64.weights[1], 2) == 25
    @test all(size(p64.weights[i], 1) == length(p64.biases[i]) for i in eachindex(p64.weights))

    p32 = NNFoil.NeuralNetworkParameters(; model_size = :xsmall, T = Float32)
    @test eltype(p32.mean_inputs_scaled) == Float32
    @test eltype(p32.weights[1]) == Float32
end

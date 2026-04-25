@testset "output container shapes" begin
    out_vector = NNFoil.NeuralNetworkOutput(
        analysis_confidence = [0.9, 0.8],
        CL = [0.1, 0.2],
        CD = [0.01, 0.02],
        CM = [-0.05, -0.04],
        Top_Xtr = [0.7, 0.6],
        Bot_Xtr = [0.5, 0.4],
    )
    @test out_vector.CL isa Vector{Float64}

    out_scalar = NNFoil.NeuralNetworkOutput(
        analysis_confidence = 0.9,
        CL = 0.1,
        CD = 0.01,
        CM = -0.05,
        Top_Xtr = 0.7,
        Bot_Xtr = 0.5,
    )
    @test out_scalar.CL isa Float64

    @test_throws MethodError NNFoil.NeuralNetworkOutput(
        analysis_confidence = 0.9,
        CL = [0.1],
        CD = [0.01],
        CM = [-0.05],
        Top_Xtr = [0.7],
        Bot_Xtr = [0.5],
    )
end

@testset "match input/output types" begin
    x_scalar = NNFoil.build_features(UNIT_KULFAN, 2.0, UNIT_REYNOLDS_SCALAR)
    out_scalar = NNFoil.evaluate(UNIT_NETWORK_PARAMETERS, x_scalar)

    @test out_scalar.analysis_confidence isa eltype(x_scalar)
    @test out_scalar.CL isa eltype(x_scalar)
    @test out_scalar.CD isa eltype(x_scalar)
    @test out_scalar.CM isa eltype(x_scalar)
    @test out_scalar.Top_Xtr isa eltype(x_scalar)
    @test out_scalar.Bot_Xtr isa eltype(x_scalar)

    out_matrix = NNFoil.evaluate(UNIT_NETWORK_PARAMETERS, reshape(x_scalar, :, 1))
    @test out_matrix.CL isa AbstractVector
end

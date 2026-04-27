@testset "allocate_forward_cache" begin
    x_matrix = NNFoil.build_features(UNIT_KULFAN, UNIT_ALPHA[1:3], UNIT_REYNOLDS_SCALAR)
    y_matrix, tmp_matrix = NNFoil.allocate_forward_cache(UNIT_NETWORK_PARAMETERS, x_matrix)

    @test size(y_matrix) == (size(UNIT_NETWORK_PARAMETERS.weights[end], 1), 3)
    @test length(tmp_matrix) == length(UNIT_NETWORK_PARAMETERS.weights) + 1
    @test tmp_matrix[1] === x_matrix
    @test all(size(tmp_matrix[i], 2) == 3 for i in 2:length(tmp_matrix))

    x_vector = NNFoil.build_features(UNIT_KULFAN, 2.0, 5.0e6)
    y_vector, tmp_vector = NNFoil.allocate_forward_cache(UNIT_NETWORK_PARAMETERS, x_vector)

    @test size(y_vector) == (size(UNIT_NETWORK_PARAMETERS.weights[end], 1), 1)
    @test tmp_vector[1] === x_vector
    @test ndims(tmp_vector[2]) == 1

    @testset "Float32" begin
        params32 = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float32)
        kulfan32 = NNFoil.KulfanParameters(
            upper_weights = Float32.(UNIT_KULFAN.upper_weights),
            lower_weights = Float32.(UNIT_KULFAN.lower_weights),
            leading_edge_weight = Float32(UNIT_KULFAN.leading_edge_weight),
            trailing_edge_thickness = Float32(UNIT_KULFAN.trailing_edge_thickness),
        )
        x32 = NNFoil.build_features(kulfan32, Float32.(UNIT_ALPHA[1:3]), Float32(UNIT_REYNOLDS_SCALAR))
        y32, tmp32 = NNFoil.allocate_forward_cache(params32, x32)

        @test eltype(y32) == Float32
        @test eltype(tmp32[1]) == Float32
        @test eltype(tmp32[2]) == Float32
    end
end

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
end

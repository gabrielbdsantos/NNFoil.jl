@testset "normalize_coordinates!" begin
    @testset "chord alignment and similarity invariance" begin
        reference = [1.0 0.01; 0.5 0.1; 0.0 0.0; 0.5 -0.05; 1.0 -0.01]
        for angle in (0.0, 0.7, 2.0)
            rotation = [cos(angle) -sin(angle); sin(angle) cos(angle)]
            coords = 3 .* reference * rotation' .+ [2.0 -4.0]
            @test NNFoil.normalize_coordinates!(coords) === coords
            @test coords ≈ reference atol = 1.0e-14
        end
    end

    @testset "negative-x geometric leading edge" begin
        coords = [1.0 0.01; 0.5 0.1; -0.0001 0.002; 0.0 0.0; 0.5 -0.05; 1.0 -0.01]
        normalized = NNFoil.normalize_coordinates!(copy(coords))
        @test normalized[3, :] == [0.0, 0.0]
        @test (normalized[1, :] + normalized[end, :]) / 2 ≈ [1.0, 0.0] atol = 1.0e-14
        @test normalized[4, :] != [0.0, 0.0]
        @test NNFoil.normalize_coordinates!(copy(normalized)) ≈ normalized atol = 1.0e-14
    end
end

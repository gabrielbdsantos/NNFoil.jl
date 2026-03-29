@testset "split_upper_lower_surfaces" begin
    coords_odd = [
        1.0 0.0
        0.5 0.2
        0.0 0.0
        0.5 -0.2
        1.0 0.0
    ]
    upper_odd, lower_odd = NNFoil.split_upper_lower_surfaces(coords_odd)
    @test size(upper_odd) == (3, 2)
    @test size(lower_odd) == (3, 2)
    @test collect(upper_odd[end, :]) == collect(coords_odd[3, :])
    @test collect(lower_odd[1, :]) == collect(coords_odd[3, :])

    coords_even = [
        1.0 0.0
        0.5 0.2
        0.0 0.0
        0.4 -0.1
        0.8 -0.05
        1.0 0.0
    ]
    upper_even, lower_even = NNFoil.split_upper_lower_surfaces(coords_even)
    @test size(upper_even) == (3, 2)
    @test size(lower_even) == (3, 2)
    @test collect(upper_even[end, :]) == collect(coords_even[3, :])
    @test collect(lower_even[1, :]) == collect(coords_even[4, :])
end

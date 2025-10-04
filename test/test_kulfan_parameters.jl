@testset "Kulfan Parameters ($database)" for database in readdir(AIRFOILS_DIR; join=true)
    @testset "$filename" for filename in readdir(database; join=true)
        coords = coordinates_from_file(filename)
        py_ans = convert_kulfan_py2jl(
            py_get_kulfan_from_coordinates(py_array(coords); normalize_coordinates=false)
        )
        jl_ans = NNFoil.get_kulfan_parameters(coords)

        # NOTE: an absolute tolerance of 1e-6 is enought to get consistent results out
        # of the neural network.
        @test isapprox(py_ans, jl_ans; atol=1e-6)
    end
end

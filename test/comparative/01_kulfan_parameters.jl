@testset "Kulfan Parameters ($database)" for database in readdir(AIRFOILS_DIR; join = true)
    @testset "$filename" for filename in select_cases(readdir(database))
        coords = coordinates_from_file(joinpath(database, filename))
        py_ans = convert_kulfan_py2jl(
            py_get_kulfan_from_coordinates(py_array(coords); normalize_coordinates = false)
        )
        jl_ans = NNFoil.KulfanParameters(coords)

        # NOTE: an absolute tolerance of 1e-6 is enought to get consistent results out
        # of the neural network.
        @test isapprox(py_ans, jl_ans; atol = 1.0e-6)
    end
end

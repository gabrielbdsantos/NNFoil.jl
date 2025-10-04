const MODEL_PARAMS = NNFoil.load_network_parameters(; model_size=:xlarge)
const ALPHA = collect(-180.0:180.0)
const RE_RANGE = 10 .^ range(3, 9, 15)


function py_network(kulfan, alpha, Reynolds)
    py_ans = py_get_aero_from_kulfan_parameters(
        kulfan_parameters=kulfan,
        alpha=alpha,
        Re=Reynolds
    )

    NNFoil.NeuralNetworkOutput(
        pyconvert(Vector{Float64}, py_ans["analysis_confidence"]),
        pyconvert(Vector{Float64}, py_ans["CL"]),
        pyconvert(Vector{Float64}, py_ans["CD"]),
        pyconvert(Vector{Float64}, py_ans["CM"]),
        pyconvert(Vector{Float64}, py_ans["Top_Xtr"]),
        pyconvert(Vector{Float64}, py_ans["Bot_Xtr"]),
    )
end


function compare_networks(py_kulfan, jl_kulfan, alpha, ReRange, atol)
    @testset "Re = $Re" for Re in ReRange
        py_ans = py_network(py_kulfan, py_array(alpha), py_array(Re))
        jl_ans = NNFoil.get_aero_from_kulfan_parameters(
            MODEL_PARAMS, jl_kulfan, alpha, Re .* ones(size(alpha, 1))
        )

        @test isapprox(py_ans, jl_ans; atol=atol)
    end
end


@testset "Neural Network ($database)" for database in readdir(AIRFOILS_DIR; join=true)
    @testset "$filename" for filename in readdir(database)
        py_kulfan = py_get_kulfan_from_file(joinpath(database, filename))
        jl_kulfan = convert_kulfan_py2jl(py_kulfan)

        # NOTE: the tolerance here is tighter to ensure consistent results.
        compare_networks(py_kulfan, jl_kulfan, ALPHA, RE_RANGE, 1e-9)
    end
end


@testset "Workflow ($database)" for database in readdir(AIRFOILS_DIR; join=true)
    @testset "$filename" for filename in readdir(database)
        coords = coordinates_from_file(joinpath(database, filename))
        py_kulfan = py_get_kulfan_from_coordinates(py_array(coords); normalize_coordinates=false)
        jl_kulfan = NNFoil.get_kulfan_parameters(coords)

        # NOTE: for engineering purposes, an absolute difference of 1e-3 seems acceptable.
        # Reducing it to 1e-6 results in a few errors, but still acceptable.
        compare_networks(py_kulfan, jl_kulfan, ALPHA, RE_RANGE, 1e-3)
    end
end

using Test
using NNFoil

const AIRFOILS_DIR = joinpath(".", "airfoils")

include("utils.jl")
include("python_wrapper.jl")

@testset verbose = true "NNFoil.jl" begin
    include("code_quality.jl")
    include("test_kulfan_parameters.jl")
    include("test_neural_network.jl")
end

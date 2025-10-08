using Test
using NNFoil

import StatsBase

const AIRFOILS_DIR = joinpath(".", "airfoils")
const CHOOSE_RANDOMLY = false  # "CI" in ARGS ? true : false
const LIMIT_NUM_CASES = 250
const NUM_REYNOLDS_VALUES = 15

# NOTE: all tests use the "xsmall" network simply because it is faster. Using a
# larger network size is not expected to produce a different outcome than what
# is already obtained.
const MODEL_SIZE = :xsmall

function select_cases(cases::AbstractVector{<:AbstractString})
    if CHOOSE_RANDOMLY === true
        return StatsBase.sample(cases, LIMIT_NUM_CASES)
    else
        return cases
    end
end

include("utils.jl")
include("python_wrapper.jl")

@testset verbose = true "NNFoil.jl" begin
    @testset "Code analysis" begin
        include("code_quality.jl")
    end

    @testset "Unit tests" begin
        # Nothing yet
    end

    @testset verbose = true "Comparison against NeuralFoil (Python)" begin
        include("test_kulfan_parameters.jl")
        include("test_neural_network.jl")
    end
end

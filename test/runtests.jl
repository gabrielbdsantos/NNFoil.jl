using Test
using NNFoil
using DelimitedFiles

const AIRFOILS_DIR = joinpath(".", "airfoils")
const CHOOSE_RANDOMLY = false  # "CI" in ARGS ? true : false
const LIMIT_NUM_CASES = 250
const NUM_REYNOLDS_VALUES = 15

# Only `import StatsBase` if needed.
CHOOSE_RANDOMLY ? (import StatsBase) : nothing

# NOTE: all tests use the "xsmall" network simply because it is faster. Using a
# larger network size is not expected to produce a different outcome.
const MODEL_SIZE = :xsmall

function select_cases(cases::AbstractVector{<:AbstractString})
    if CHOOSE_RANDOMLY === true
        return StatsBase.sample(cases, LIMIT_NUM_CASES)
    else
        return cases
    end
end

include("utils.jl")

@testset verbose = true "NNFoil.jl" begin
    @testset "Code analysis" begin
        import Aqua
        import JET

        @testset "Code quality (Aqua.jl)" begin
            Aqua.test_all(NNFoil)
        end

        @testset "Code linting (JET.jl)" begin
            JET.test_package(NNFoil; target_defined_modules = true)
        end
    end

    @testset "Unit tests" begin
        for file in readdir(joinpath(@__DIR__, "unittests"); join = true, sort = true)
            endswith(file, ".jl") || continue
            include(file)
        end
    end

    @testset verbose = true "Comparison against NeuralFoil (Python)" begin
        for file in readdir(joinpath(@__DIR__, "comparative"); join = true, sort = true)
            endswith(file, ".jl") || continue
            include(file)
        end
    end
end

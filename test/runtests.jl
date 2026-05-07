using Test
using NNFoil
using DelimitedFiles

# NOTE: all tests use the "xsmall" network simply because it is faster. Using a
# larger network size is not expected to produce a different outcome.
const MODEL_SIZE = :xsmall
const AIRFOILS_DIR = joinpath(".", "airfoils")
const NUM_REYNOLDS_VALUES = 15

# Test arguments:
#   no-code-quality     : disables code quality tests
#   no-unit-tests       : disables unit tests
#   no-comparison       : disables comparative tests
#   reduced-comparison  : reduces comparative tests to a random set of N cases
const TEST_ARGS = Set(lowercase.(ARGS))
const RUN_CODE_QUALITY = !("no-code-quality" in TEST_ARGS)
const RUN_UNIT_TESTS = !("no-unit-tests" in TEST_ARGS)
const RUN_COMPARATIVE_SET = !("no-comparison" in TEST_ARGS)
const RUN_REDUCED_COMPARATIVE_SET = !("full" in TEST_ARGS)
const CHOOSE_N_CASES_RANDOMLY = 100

# Only `import StatsBase` if needed.
RUN_REDUCED_COMPARATIVE_SET ? (import StatsBase) : nothing

function select_cases(cases)
    if RUN_REDUCED_COMPARATIVE_SET === true
        return StatsBase.sample(cases, CHOOSE_N_CASES_RANDOMLY)
    else
        return cases
    end
end

include("utils.jl")

@testset failfast = true verbose = true "NNFoil.jl" begin
    if RUN_CODE_QUALITY
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
    end

    if RUN_UNIT_TESTS
        @testset "Unit tests" begin
            for file in readdir(joinpath(@__DIR__, "unittests"); join = true, sort = true)
                endswith(file, ".jl") && include(file)
            end
        end
    end

    if RUN_COMPARATIVE_SET
        @testset verbose = true "Comparison against NeuralFoil (Python)" begin
            for file in readdir(joinpath(@__DIR__, "comparative"); join = true, sort = true)
                endswith(file, ".jl") && include(file)
            end
        end
    end
end

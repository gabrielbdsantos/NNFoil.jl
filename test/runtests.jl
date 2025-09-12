using NNFoil
using Test
using Aqua
using JET

@testset "NNFoil.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(NNFoil)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(NNFoil; target_defined_modules = true)
    end
    # Write your tests here.
end

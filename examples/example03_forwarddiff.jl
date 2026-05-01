import Pkg

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const TEST_DIR = joinpath(ROOT_DIR, "test")

Pkg.activate(TEST_DIR; io = devnull)
pushfirst!(LOAD_PATH, ROOT_DIR)

using NNFoil
using DelimitedFiles
using ForwardDiff
using FiniteDiff
using LinearAlgebra

function objective_evaluate(x_vec, params, n_cols)
    x = reshape(x_vec, 25, n_cols)
    out = evaluate(params, x)
    return sum(out.CL ./ out.CD)
end

function objective_evaluate_inplace(x_vec, params, n_cols)
    x = reshape(x_vec, 25, n_cols)
    cache = NeuralNetworkCache(params, x)
    evaluate!(cache)
    return sum(cache.outputs.CL ./ cache.outputs.CD)
end

function relative_error(a, b)
    denom = max(norm(a), eps(eltype(a)))
    return norm(a - b) / denom
end

coordinates = readdlm(
    abspath(
        joinpath(
            NNFoil.DATA_PATH, splitpath("../test/airfoils/raw/naca0018.dat")...
        )
    )
)

kulfan_parameters = KulfanParameters(normalize_coordinates!(coordinates))
network_parameters = NeuralNetworkParameters(; model_size = :xsmall, T = Float64)

alpha = [-2.0, 3.0, 7.0]
Reynolds = 5.0e6

x = build_features(kulfan_parameters, alpha, Reynolds)
x_vec = vec(copy(x))
n_cols = size(x, 2)

g_forwarddiff_evaluate = ForwardDiff.gradient(
    v -> objective_evaluate(v, network_parameters, n_cols),
    x_vec,
)
g_finitediff_evaluate = FiniteDiff.finite_difference_gradient(
    v -> objective_evaluate(v, network_parameters, n_cols),
    x_vec,
)

g_forwarddiff_evaluate_inplace = ForwardDiff.gradient(
    v -> objective_evaluate_inplace(v, network_parameters, n_cols),
    x_vec,
)
g_finitediff_evaluate_inplace = FiniteDiff.finite_difference_gradient(
    v -> objective_evaluate_inplace(v, network_parameters, n_cols),
    x_vec,
)

err_evaluate = relative_error(g_forwarddiff_evaluate, g_finitediff_evaluate)
err_evaluate_inplace = relative_error(
    g_forwarddiff_evaluate_inplace,
    g_finitediff_evaluate_inplace,
)

const tol = 5.0e-8

println("ForwardDiff vs FiniteDiff on evaluate(params, x):")
println("  relative gradient error = $(err_evaluate)")
println("  within tolerance ($(tol)) = $(err_evaluate <= tol)")

println("\nForwardDiff vs FiniteDiff on evaluate!(cache):")
println("  relative gradient error = $(err_evaluate_inplace)")
println("  within tolerance ($(tol)) = $(err_evaluate_inplace <= tol)")

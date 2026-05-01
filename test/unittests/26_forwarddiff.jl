using ForwardDiff
using FiniteDiff

function _ad_objective_evaluate(x_vec, params, n_cols)
    x = reshape(x_vec, 25, n_cols)
    out = NNFoil.evaluate(params, x)
    return sum(out.CL ./ out.CD)
end

function _ad_objective_evaluate_inplace(x_vec, params, n_cols)
    x = reshape(x_vec, 25, n_cols)
    cache = NNFoil.NeuralNetworkCache(params, x)
    NNFoil.evaluate!(cache)
    return sum(cache.outputs.CL ./ cache.outputs.CD)
end

function _forwarddiff_gradient_evaluate(params, kulfan, alpha, Reynolds)
    x = NNFoil.build_features(kulfan, alpha, Reynolds)
    x_vec = vec(copy(x))
    n_cols = size(x, 2)
    g = ForwardDiff.gradient(v -> _ad_objective_evaluate(v, params, n_cols), x_vec)

    return g, x_vec
end

function _forwarddiff_gradient_evaluate_inplace(params, kulfan, alpha, Reynolds)
    x = NNFoil.build_features(kulfan, alpha, Reynolds)
    x_vec = vec(copy(x))
    n_cols = size(x, 2)
    g = ForwardDiff.gradient(v -> _ad_objective_evaluate_inplace(v, params, n_cols), x_vec)

    return g, x_vec
end

function _finitediff_gradient(f, x_vec)
    return FiniteDiff.finite_difference_gradient(f, x_vec)
end

function _forwarddiff_kulfan_to_vector(kulfan::NNFoil.KulfanParameters)
    T = promote_type(
        eltype(kulfan.upper_weights),
        eltype(kulfan.lower_weights),
        typeof(kulfan.leading_edge_weight),
        typeof(kulfan.trailing_edge_thickness),
    )
    v = Vector{T}(undef, 18)
    v[1:8] .= kulfan.upper_weights
    v[9:16] .= kulfan.lower_weights
    v[17] = kulfan.leading_edge_weight
    v[18] = kulfan.trailing_edge_thickness
    return v
end

function _forwarddiff_kulfan_from_vector(v::AbstractVector{<:Real})
    return NNFoil.KulfanParameters(
        upper_weights = v[1:8],
        lower_weights = v[9:16],
        leading_edge_weight = v[17],
        trailing_edge_thickness = v[18],
    )
end

function _ad_objective_kulfan_evaluate(k_vec, params, alpha, Reynolds)
    kulfan = _forwarddiff_kulfan_from_vector(k_vec)
    x = NNFoil.build_features(kulfan, alpha, Reynolds)
    out = NNFoil.evaluate(params, x)
    return sum(out.CL ./ out.CD)
end

function _ad_objective_kulfan_evaluate_inplace(k_vec, params, alpha, Reynolds)
    kulfan = _forwarddiff_kulfan_from_vector(k_vec)
    x = NNFoil.build_features(kulfan, alpha, Reynolds)
    cache = NNFoil.NeuralNetworkCache(params, x)
    NNFoil.evaluate!(cache)
    return sum(cache.outputs.CL ./ cache.outputs.CD)
end

function _forwarddiff_gradient_kulfan_evaluate(params, kulfan, alpha, Reynolds)
    k_vec = _forwarddiff_kulfan_to_vector(kulfan)
    g = ForwardDiff.gradient(v -> _ad_objective_kulfan_evaluate(v, params, alpha, Reynolds), k_vec)
    return g, k_vec
end

function _forwarddiff_gradient_kulfan_evaluate_inplace(params, kulfan, alpha, Reynolds)
    k_vec = _forwarddiff_kulfan_to_vector(kulfan)
    g = ForwardDiff.gradient(
        v -> _ad_objective_kulfan_evaluate_inplace(v, params, alpha, Reynolds),
        k_vec,
    )
    return g, k_vec
end

@testset "ForwardDiff" begin
    @testset "evaluate(params, x)" begin
        @testset "Float64" begin
            params64 = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float64)
            n_cols = 3
            g64, xvec64 = _forwarddiff_gradient_evaluate(
                params64,
                UNIT_KULFAN,
                UNIT_ALPHA[1:n_cols],
                UNIT_REYNOLDS_SCALAR,
            )
            g64_fd = _finitediff_gradient(
                v -> _ad_objective_evaluate(v, params64, n_cols),
                xvec64,
            )

            @test eltype(g64) == Float64
            @test length(g64) == length(xvec64)
            @test all(isfinite, g64)
            @test g64 ≈ g64_fd rtol = 2.0e-6 atol = 1.0e-8
        end

        @testset "Float32" begin
            kulfan32 = NNFoil.KulfanParameters(
                upper_weights = Float32.(UNIT_KULFAN.upper_weights),
                lower_weights = Float32.(UNIT_KULFAN.lower_weights),
                leading_edge_weight = Float32(UNIT_KULFAN.leading_edge_weight),
                trailing_edge_thickness = Float32(UNIT_KULFAN.trailing_edge_thickness),
            )
            alpha32 = Float32.(UNIT_ALPHA[1:3])
            Reynolds32 = Float32(UNIT_REYNOLDS_SCALAR)
            params32 = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float32)
            n_cols = length(alpha32)

            g32, xvec32 = _forwarddiff_gradient_evaluate(params32, kulfan32, alpha32, Reynolds32)
            g32_fd = _finitediff_gradient(
                v -> _ad_objective_evaluate(v, params32, n_cols),
                xvec32,
            )

            @test eltype(g32) == Float32
            @test length(g32) == length(xvec32)
            @test all(isfinite, g32)
            @test g32 ≈ g32_fd rtol = 2.0e-2 atol = 2.0e-3

            x32_seed = copy(NNFoil.build_features(kulfan32, alpha32, Reynolds32))

            function f32_single_feature(a)
                x = Matrix{typeof(a)}(x32_seed)
                x[19, 1] = a
                out = NNFoil.evaluate(params32, x)
                return out.CL[1] / out.CD[1]
            end

            d32 = ForwardDiff.derivative(f32_single_feature, x32_seed[19, 1])
            d32_fd = FiniteDiff.finite_difference_derivative(
                f32_single_feature,
                x32_seed[19, 1],
            )
            @test isfinite(d32)
            @test d32 isa Float32
            @test d32 ≈ d32_fd rtol = 5.0e-4 atol = 1.0e-5
        end
    end

    @testset "evaluate!(cache)" begin
        @testset "Float64" begin
            params64 = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float64)
            n_cols = 3

            g64_eval, xvec64_eval = _forwarddiff_gradient_evaluate(
                params64,
                UNIT_KULFAN,
                UNIT_ALPHA[1:n_cols],
                UNIT_REYNOLDS_SCALAR,
            )
            g64_inplace, xvec64_inplace = _forwarddiff_gradient_evaluate_inplace(
                params64,
                UNIT_KULFAN,
                UNIT_ALPHA[1:n_cols],
                UNIT_REYNOLDS_SCALAR,
            )
            g64_fd = _finitediff_gradient(
                v -> _ad_objective_evaluate_inplace(v, params64, n_cols),
                xvec64_inplace,
            )

            @test eltype(g64_inplace) == Float64
            @test length(g64_inplace) == length(xvec64_inplace)
            @test all(isfinite, g64_inplace)
            @test xvec64_inplace == xvec64_eval
            @test g64_inplace ≈ g64_eval
            @test g64_inplace ≈ g64_fd rtol = 2.0e-6 atol = 1.0e-8
        end

        @testset "Float32" begin
            kulfan32 = NNFoil.KulfanParameters(
                upper_weights = Float32.(UNIT_KULFAN.upper_weights),
                lower_weights = Float32.(UNIT_KULFAN.lower_weights),
                leading_edge_weight = Float32(UNIT_KULFAN.leading_edge_weight),
                trailing_edge_thickness = Float32(UNIT_KULFAN.trailing_edge_thickness),
            )
            alpha32 = Float32.(UNIT_ALPHA[1:3])
            Reynolds32 = Float32(UNIT_REYNOLDS_SCALAR)
            params32 = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float32)
            n_cols = length(alpha32)

            g32_eval, xvec32_eval = _forwarddiff_gradient_evaluate(
                params32,
                kulfan32,
                alpha32,
                Reynolds32,
            )
            g32_inplace, xvec32_inplace = _forwarddiff_gradient_evaluate_inplace(
                params32,
                kulfan32,
                alpha32,
                Reynolds32,
            )
            g32_fd = _finitediff_gradient(
                v -> _ad_objective_evaluate_inplace(v, params32, n_cols),
                xvec32_inplace,
            )

            @test eltype(g32_inplace) == Float32
            @test length(g32_inplace) == length(xvec32_inplace)
            @test all(isfinite, g32_inplace)
            @test xvec32_inplace == xvec32_eval
            @test g32_inplace ≈ g32_eval
            @test g32_inplace ≈ g32_fd rtol = 2.0e-2 atol = 2.0e-3

            x32_seed = copy(NNFoil.build_features(kulfan32, alpha32, Reynolds32))

            function f32_single_feature_inplace(a)
                x = Matrix{typeof(a)}(x32_seed)
                x[19, 1] = a
                cache = NNFoil.NeuralNetworkCache(params32, x)
                NNFoil.evaluate!(cache)
                return cache.outputs.CL[1] / cache.outputs.CD[1]
            end

            d32_in = ForwardDiff.derivative(f32_single_feature_inplace, x32_seed[19, 1])
            d32_in_fd = FiniteDiff.finite_difference_derivative(
                f32_single_feature_inplace,
                x32_seed[19, 1],
            )
            @test isfinite(d32_in)
            @test d32_in isa Float32
            @test d32_in ≈ d32_in_fd rtol = 5.0e-4 atol = 1.0e-5

            function f32_single_feature(a)
                x = Matrix{typeof(a)}(x32_seed)
                x[19, 1] = a
                out = NNFoil.evaluate(params32, x)
                return out.CL[1] / out.CD[1]
            end

            d32 = ForwardDiff.derivative(f32_single_feature, x32_seed[19, 1])
            @test d32_in ≈ d32
        end
    end

    @testset "wrt Kulfan parameters" begin
        @testset "evaluate(params, x)" begin
            @testset "Float64" begin
                params64 = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float64)
                alpha64 = UNIT_ALPHA[1:1]
                Reynolds64 = UNIT_REYNOLDS_SCALAR

                g64, kvec64 = _forwarddiff_gradient_kulfan_evaluate(
                    params64,
                    UNIT_KULFAN,
                    alpha64,
                    Reynolds64,
                )
                g64_fd = _finitediff_gradient(
                    v -> _ad_objective_kulfan_evaluate(v, params64, alpha64, Reynolds64),
                    kvec64,
                )

                @test eltype(g64) == Float64
                @test length(g64) == length(kvec64)
                @test all(isfinite, g64)
                @test g64 ≈ g64_fd rtol = 1.0e-5 atol = 1.0e-8
            end

            @testset "Float32" begin
                kulfan32 = NNFoil.KulfanParameters(
                    upper_weights = Float32.(UNIT_KULFAN.upper_weights),
                    lower_weights = Float32.(UNIT_KULFAN.lower_weights),
                    leading_edge_weight = Float32(UNIT_KULFAN.leading_edge_weight),
                    trailing_edge_thickness = Float32(UNIT_KULFAN.trailing_edge_thickness),
                )
                alpha32 = Float32.(UNIT_ALPHA[1:1])
                Reynolds32 = Float32(UNIT_REYNOLDS_SCALAR)
                params32 = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float32)

                g32, kvec32 = _forwarddiff_gradient_kulfan_evaluate(
                    params32,
                    kulfan32,
                    alpha32,
                    Reynolds32,
                )
                # NOTE: Compute the finite-difference reference in Float64 to
                # reduce numerical noise from finite-difference stencils while
                # still checking the Float32 AD gradient against the same
                # objective definition. This stabilization is currently specific
                # to the ForwardDiff Kulfan-parameter checks.
                g32_fd = Float32.(
                    _finitediff_gradient(
                        v -> _ad_objective_kulfan_evaluate(v, params32, alpha32, Reynolds32),
                        Float64.(kvec32),
                    )
                )

                @test eltype(g32) == Float32
                @test length(g32) == length(kvec32)
                @test all(isfinite, g32)
                @test g32 ≈ g32_fd rtol = 5.0e-3 atol = 1.0e-3
            end
        end

        @testset "evaluate!(cache)" begin
            @testset "Float64" begin
                params64 = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float64)
                alpha64 = UNIT_ALPHA[1:1]
                Reynolds64 = UNIT_REYNOLDS_SCALAR

                g64_eval, kvec64_eval = _forwarddiff_gradient_kulfan_evaluate(
                    params64,
                    UNIT_KULFAN,
                    alpha64,
                    Reynolds64,
                )
                g64_inplace, kvec64_inplace = _forwarddiff_gradient_kulfan_evaluate_inplace(
                    params64,
                    UNIT_KULFAN,
                    alpha64,
                    Reynolds64,
                )
                g64_fd = _finitediff_gradient(
                    v -> _ad_objective_kulfan_evaluate_inplace(v, params64, alpha64, Reynolds64),
                    kvec64_inplace,
                )

                @test eltype(g64_inplace) == Float64
                @test length(g64_inplace) == length(kvec64_inplace)
                @test all(isfinite, g64_inplace)
                @test kvec64_inplace == kvec64_eval
                @test g64_inplace ≈ g64_eval
                @test g64_inplace ≈ g64_fd rtol = 1.0e-5 atol = 1.0e-8
            end

            @testset "Float32" begin
                kulfan32 = NNFoil.KulfanParameters(
                    upper_weights = Float32.(UNIT_KULFAN.upper_weights),
                    lower_weights = Float32.(UNIT_KULFAN.lower_weights),
                    leading_edge_weight = Float32(UNIT_KULFAN.leading_edge_weight),
                    trailing_edge_thickness = Float32(UNIT_KULFAN.trailing_edge_thickness),
                )
                alpha32 = Float32.(UNIT_ALPHA[1:1])
                Reynolds32 = Float32(UNIT_REYNOLDS_SCALAR)
                params32 = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float32)

                g32_eval, kvec32_eval = _forwarddiff_gradient_kulfan_evaluate(
                    params32,
                    kulfan32,
                    alpha32,
                    Reynolds32,
                )
                g32_inplace, kvec32_inplace = _forwarddiff_gradient_kulfan_evaluate_inplace(
                    params32,
                    kulfan32,
                    alpha32,
                    Reynolds32,
                )

                # NOTE: Compute the finite-difference reference in Float64 to
                # reduce numerical noise from finite-difference stencils while
                # still checking the Float32 AD gradient against the same
                # objective definition. This stabilization is currently specific
                # to the ForwardDiff Kulfan-parameter checks.
                g32_fd = Float32.(
                    _finitediff_gradient(
                        v -> _ad_objective_kulfan_evaluate_inplace(v, params32, alpha32, Reynolds32),
                        Float64.(kvec32_inplace),
                    )
                )

                @test eltype(g32_inplace) == Float32
                @test length(g32_inplace) == length(kvec32_inplace)
                @test all(isfinite, g32_inplace)
                @test kvec32_inplace == kvec32_eval
                @test g32_inplace ≈ g32_eval
                @test g32_inplace ≈ g32_fd rtol = 5.0e-3 atol = 1.0e-3
            end
        end
    end
end

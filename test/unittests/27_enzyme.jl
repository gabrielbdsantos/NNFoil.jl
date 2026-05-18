using Enzyme

function _enzyme_objective_evaluate(x, params)
    out = NNFoil.evaluate(params, x)
    return out.CL / out.CD
end

function _enzyme_objective_evaluate_inplace(x, cache)
    copyto!(cache.x, x)
    copyto!(cache.x_flipped, x)
    NNFoil.flip_inputs!(cache.x_flipped)
    NNFoil.evaluate!(cache)
    return cache.outputs.CL[1] / cache.outputs.CD[1]
end

function _enzyme_gradient(f, x)
    dx = zero(x)
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
    Enzyme.autodiff(mode, Enzyme.Const(f), Enzyme.Active, Enzyme.Duplicated(x, dx))
    return dx
end

function _enzyme_gradient(f, x, cache)
    dx = zero(x)
    dcache = _enzyme_zero_cache(cache)
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
    Enzyme.autodiff(
        mode,
        Enzyme.Const(f),
        Enzyme.Active,
        Enzyme.Duplicated(x, dx),
        Enzyme.Duplicated(cache, dcache),
    )
    return dx
end

function _enzyme_zero_cache(cache)
    x0 = cache.x isa AbstractVector ? zero(cache.x) : zero(cache.x_both)
    dcache = NNFoil.NeuralNetworkCache(cache.network_parameters, x0)
    _enzyme_zero!(dcache)
    return dcache
end

function _enzyme_zero!(x)
    return x
end

function _enzyme_zero!(x::AbstractArray{<:Real})
    fill!(x, zero(eltype(x)))
    return x
end

function _enzyme_zero!(x::AbstractArray)
    foreach(_enzyme_zero!, x)
    return x
end

function _enzyme_zero!(x::NNFoil.NeuralNetworkOutput)
    foreach(name -> _enzyme_zero!(getfield(x, name)), fieldnames(typeof(x)))
    return x
end

function _enzyme_zero!(cache::NNFoil.NeuralNetworkCache)
    foreach(name -> _enzyme_zero!(getfield(cache, name)), fieldnames(typeof(cache)))
    return cache
end

@testset "Enzyme" begin
    @testset "evaluate(params, x)" begin
        @testset "Float64" begin
            params = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float64)
            x = NNFoil.build_features(UNIT_KULFAN, UNIT_ALPHA[1], UNIT_REYNOLDS_SCALAR)

            g = _enzyme_gradient(
                v -> _enzyme_objective_evaluate(v, params),
                x,
            )
            g_fd = FiniteDiff.finite_difference_gradient(
                v -> _enzyme_objective_evaluate(v, params),
                x,
            )

            @test eltype(g) == Float64
            @test length(g) == length(x)
            @test all(isfinite, g)
            @test g ≈ g_fd rtol = 2.0e-6 atol = 1.0e-8
        end

        @testset "Float32" begin
            kulfan32 = NNFoil.KulfanParameters(
                upper_weights = Float32.(UNIT_KULFAN.upper_weights),
                lower_weights = Float32.(UNIT_KULFAN.lower_weights),
                leading_edge_weight = Float32(UNIT_KULFAN.leading_edge_weight),
                trailing_edge_thickness = Float32(UNIT_KULFAN.trailing_edge_thickness),
            )
            alpha32 = Float32(UNIT_ALPHA[1])
            Reynolds32 = Float32(UNIT_REYNOLDS_SCALAR)
            params = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float32)
            x = NNFoil.build_features(kulfan32, alpha32, Reynolds32)

            g = _enzyme_gradient(
                v -> _enzyme_objective_evaluate(v, params),
                x,
            )
            g_fd = FiniteDiff.finite_difference_gradient(
                v -> _enzyme_objective_evaluate(v, params),
                x,
            )

            @test eltype(g) == Float32
            @test length(g) == length(x)
            @test all(isfinite, g)
            @test g ≈ g_fd rtol = 2.0e-2 atol = 2.0e-3
        end
    end

    @testset "evaluate!(cache)" begin
        @testset "Float64" begin
            params = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float64)
            x = NNFoil.build_features(UNIT_KULFAN, UNIT_ALPHA[1], UNIT_REYNOLDS_SCALAR)
            cache = NNFoil.NeuralNetworkCache(params, x)

            g_eval = _enzyme_gradient(
                v -> _enzyme_objective_evaluate(v, params),
                x,
            )
            g_inplace = _enzyme_gradient(
                _enzyme_objective_evaluate_inplace,
                x,
                cache,
            )
            g_fd = FiniteDiff.finite_difference_gradient(
                v -> _enzyme_objective_evaluate_inplace(v, cache),
                x,
            )

            @test eltype(g_inplace) == Float64
            @test length(g_inplace) == length(x)
            @test all(isfinite, g_inplace)
            @test g_inplace ≈ g_eval
            @test g_inplace ≈ g_fd rtol = 2.0e-6 atol = 1.0e-8
        end

        @testset "Float32" begin
            kulfan32 = NNFoil.KulfanParameters(
                upper_weights = Float32.(UNIT_KULFAN.upper_weights),
                lower_weights = Float32.(UNIT_KULFAN.lower_weights),
                leading_edge_weight = Float32(UNIT_KULFAN.leading_edge_weight),
                trailing_edge_thickness = Float32(UNIT_KULFAN.trailing_edge_thickness),
            )
            alpha32 = Float32(UNIT_ALPHA[1])
            Reynolds32 = Float32(UNIT_REYNOLDS_SCALAR)
            params = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float32)
            x = NNFoil.build_features(kulfan32, alpha32, Reynolds32)
            cache = NNFoil.NeuralNetworkCache(params, x)

            g_eval = _enzyme_gradient(
                v -> _enzyme_objective_evaluate(v, params),
                x,
            )
            g_inplace = _enzyme_gradient(
                _enzyme_objective_evaluate_inplace,
                x,
                cache,
            )
            g_fd = FiniteDiff.finite_difference_gradient(
                v -> _enzyme_objective_evaluate_inplace(v, cache),
                x,
            )

            @test eltype(g_inplace) == Float32
            @test length(g_inplace) == length(x)
            @test all(isfinite, g_inplace)
            @test g_inplace ≈ g_eval
            @test g_inplace ≈ g_fd rtol = 2.0e-2 atol = 2.0e-3
        end
    end
end

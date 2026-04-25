using ForwardDiff

function _ad_objective(x_vec, params, n_cols)
    x = reshape(x_vec, 25, n_cols)
    out = NNFoil.evaluate(params, x)
    return sum(out.CL ./ out.CD)
end

function _forwarddiff_gradient(params, kulfan, alpha, Reynolds)
    x = NNFoil.build_features(kulfan, alpha, Reynolds)
    x_vec = vec(copy(x))
    n_cols = size(x, 2)
    g = ForwardDiff.gradient(v -> _ad_objective(v, params, n_cols), x_vec)

    return g, x_vec
end

@testset "ForwardDiff on evaluate(params, x)" begin
    @testset "Float64" begin
        params64 = NNFoil.NeuralNetworkParameters(; model_size = MODEL_SIZE, T = Float64)
        g64, xvec64 = _forwarddiff_gradient(
            params64,
            UNIT_KULFAN,
            UNIT_ALPHA[1:3],
            UNIT_REYNOLDS_SCALAR,
        )

        @test eltype(g64) == Float64
        @test length(g64) == length(xvec64)
        @test all(isfinite, g64)
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

        g32, xvec32 = _forwarddiff_gradient(params32, kulfan32, alpha32, Reynolds32)

        @test eltype(g32) == Float32
        @test length(g32) == length(xvec32)
        @test all(isfinite, g32)

        x32_seed = copy(NNFoil.build_features(kulfan32, alpha32, Reynolds32))

        function f32_single_feature(a)
            x = Matrix{typeof(a)}(x32_seed)
            x[19, 1] = a
            out = NNFoil.evaluate(params32, x)
            return out.CL[1] / out.CD[1]
        end

        d32 = ForwardDiff.derivative(f32_single_feature, x32_seed[19, 1])
        @test isfinite(d32)
        @test d32 isa Float32
    end
end

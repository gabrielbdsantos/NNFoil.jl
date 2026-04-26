using BenchmarkTools
using NNFoil

# NOTE: Benchmarks run against the 'xlarge' model to emulate a more realistic
# scenario.
const NETWORK_PARAMETERS = NNFoil.NeuralNetworkParameters(; model_size = :xlarge)
const KULFAN_PARAMETERS = NNFoil.KulfanParameters(
    upper_weights = collect(0.1:0.1:0.8),
    lower_weights = collect(-0.05:-0.05:-0.4),
    leading_edge_weight = 0.03,
    trailing_edge_thickness = 0.01,
)

function _make_case(n)
    ALPHA = n > 1 ? collect(range(-8.0, 12.0; length = n)) : [0.0]
    REYNOLDS = n > 1 ? collect(range(3.0e6, 8.0e6; length = n)) : [5.0e6]
    FEATURES = NNFoil.build_features(KULFAN_PARAMETERS, ALPHA, REYNOLDS)

    return (; ALPHA, REYNOLDS, FEATURES)
end

const CASES = (
    "N=1" => _make_case(1),
    "N=64" => _make_case(64),
    "N=512" => _make_case(512),
)

const ALPHA = collect(range(-8.0, 12.0; length = 64))
const REYNOLDS = collect(range(3.0e6, 8.0e6; length = length(ALPHA)))
const FEATURES = NNFoil.build_features(KULFAN_PARAMETERS, ALPHA, REYNOLDS)

const SUITE = BenchmarkGroup()

# End-to-end {{{
# ----------------------------------------------------------------------
SUITE["end_to_end"] = BenchmarkGroup()
SUITE["end_to_end"]["evaluate_out_of_place"] = BenchmarkGroup()
SUITE["end_to_end"]["evaluate_in_place"] = BenchmarkGroup()

for (label, case) in CASES
    SUITE["end_to_end"]["evaluate_out_of_place"][label] = @benchmarkable(
        NNFoil.evaluate(NETWORK_PARAMETERS, KULFAN_PARAMETERS, $case.ALPHA, $case.REYNOLDS),
        evals = 1,
    )

    SUITE["end_to_end"]["evaluate_in_place"][label] = @benchmarkable(
        NNFoil.evaluate!(cache),
        setup = (
            cache = NNFoil.NeuralNetworkCache(
                NETWORK_PARAMETERS,
                KULFAN_PARAMETERS,
                $case.ALPHA,
                $case.REYNOLDS
            )
        ),
        evals = 1,
    )
end

SUITE["end_to_end"]["evaluate_out_of_place"]["scalar"] = @benchmarkable(
    NNFoil.evaluate(NETWORK_PARAMETERS, KULFAN_PARAMETERS, 2.5, 5.0e6),
    evals = 1,
)

SUITE["end_to_end"]["evaluate_in_place"]["scalar"] = @benchmarkable(
    NNFoil.evaluate!(cache),
    setup = (
        cache = NNFoil.NeuralNetworkCache(NETWORK_PARAMETERS, KULFAN_PARAMETERS, 2.5, 5.0e6)
    ),
    evals = 1,
)

# }}}
# Features {{{
# ----------------------------------------------------------------------
SUITE["features"] = BenchmarkGroup()
SUITE["features"]["update_features_vector_vector"] = BenchmarkGroup()

SUITE["features"]["build_features_vector_vector"] = @benchmarkable(
    NNFoil.build_features(KULFAN_PARAMETERS, ALPHA, REYNOLDS),
    evals = 1,
)

SUITE["features"]["build_features_scalar_scalar"] = @benchmarkable(
    NNFoil.build_features(KULFAN_PARAMETERS, 2.5, 5.0e6),
    evals = 1,
)

for (label, case) in CASES
    SUITE["features"]["update_features_vector_vector"][label] = @benchmarkable(
        NNFoil.update_features!(cache, KULFAN_PARAMETERS, $case.ALPHA, $case.REYNOLDS),
        setup = (
            cache = NNFoil.NeuralNetworkCache(NETWORK_PARAMETERS, copy($case.FEATURES))
        ),
        evals = 1,
    )
end

SUITE["features"]["update_features_scalar_scalar"] = @benchmarkable(
    NNFoil.update_features!(cache, KULFAN_PARAMETERS, 2.5, 5.0e6),
    setup = (
        x_scalar = reshape(NNFoil.build_features(KULFAN_PARAMETERS, 2.5, 5.0e6), :, 1);
        cache = NNFoil.NeuralNetworkCache(NETWORK_PARAMETERS, x_scalar)
    ),
    evals = 1,
)
# }}}
# Kernels {{{
# ----------------------------------------------------------------------
SUITE["kernels"] = BenchmarkGroup()
SUITE["kernels"]["forward_out_of_place"] = BenchmarkGroup()
SUITE["kernels"]["forward_in_place"] = BenchmarkGroup()

for (label, case) in CASES
    SUITE["kernels"]["forward_out_of_place"][label] = @benchmarkable(
        NNFoil.forward(NETWORK_PARAMETERS, $case.FEATURES),
        evals = 1,
    )

    SUITE["kernels"]["forward_in_place"][label] = @benchmarkable(
        NNFoil.forward!(y, NETWORK_PARAMETERS, tmp),
        setup = (
            cache_tuple = NNFoil.allocate_forward_cache(NETWORK_PARAMETERS, copy($case.FEATURES));
            y = cache_tuple[1];
            tmp = cache_tuple[2]
        ),
        evals = 1,
    )
end

SUITE["kernels"]["squared_mahalanobis_distance_out_of_place"] = @benchmarkable(
    NNFoil.squared_mahalanobis_distance(NETWORK_PARAMETERS, FEATURES),
    evals = 1,
)

SUITE["kernels"]["squared_mahalanobis_distance_in_place"] = @benchmarkable(
    NNFoil.squared_mahalanobis_distance!(y, NETWORK_PARAMETERS, FEATURES, tmp1, tmp2),
    setup = (
        y = zeros(size(FEATURES, 2), 1);
        tmp1 = similar(FEATURES);
        tmp2 = similar(FEATURES)
    ),
    evals = 1,
)

SUITE["kernels"]["confidence_correction!"] = @benchmarkable(
    NNFoil.confidence_correction!(y, FEATURES, cache),
    setup = (
        y = NNFoil.forward(NETWORK_PARAMETERS, FEATURES);
        cache = NNFoil.NeuralNetworkCache(NETWORK_PARAMETERS, copy(FEATURES))
    ),
    evals = 1,
)

SUITE["kernels"]["decode_outputs!"] = @benchmarkable(
    NNFoil.decode_outputs!(y),
    setup = (y = NNFoil.forward(NETWORK_PARAMETERS, FEATURES)),
    evals = 1,
)

SUITE["kernels"]["flip_inputs!"] = @benchmarkable(
    NNFoil.flip_inputs!(x),
    setup = (x = copy(FEATURES)),
    evals = 1,
)

SUITE["kernels"]["flip_outputs!"] = @benchmarkable(
    NNFoil.flip_outputs!(y, tmp),
    setup = (
        y = NNFoil.forward(NETWORK_PARAMETERS, FEATURES);
        tmp = zeros(size(y, 2))
    ),
    evals = 1,
)

SUITE["kernels"]["fuse_predictions!"] = @benchmarkable(
    NNFoil.fuse_predictions!(y, y_flipped),
    setup = (
        y = NNFoil.forward(NETWORK_PARAMETERS, FEATURES);
        y_flipped = copy(y)
    ),
    evals = 1,
)

SUITE["kernels"]["pack_output"] = @benchmarkable(
    NNFoil.pack_output(y),
    setup = (y = NNFoil.forward(NETWORK_PARAMETERS, FEATURES)),
    evals = 1,
)

SUITE["kernels"]["pack_output!"] = @benchmarkable(
    NNFoil.pack_output!(output, y),
    setup = (
        y = NNFoil.forward(NETWORK_PARAMETERS, FEATURES);
        output = NNFoil.NeuralNetworkOutput(
            analysis_confidence = zeros(size(y, 2)),
            CL = zeros(size(y, 2)),
            CD = zeros(size(y, 2)),
            CM = zeros(size(y, 2)),
            Top_Xtr = zeros(size(y, 2)),
            Bot_Xtr = zeros(size(y, 2)),
        )
    ),
    evals = 1,
)
# }}}
# Pipelines {{{
# ----------------------------------------------------------------------
SUITE["pipelines"] = BenchmarkGroup()

SUITE["pipelines"]["flip_fuse_decode"] = @benchmarkable(
    begin
        NNFoil.flip_outputs!(y_flipped, tmp)
        NNFoil.fuse_predictions!(y, y_flipped)
        NNFoil.decode_outputs!(y)
    end,
    setup = (
        y = NNFoil.forward(NETWORK_PARAMETERS, FEATURES);
        y_flipped = copy(y);
        tmp = zeros(size(y, 2))
    ),
    evals = 1,
)

SUITE["pipelines"]["forward_two_pass_vs_concat"] = BenchmarkGroup()
SUITE["pipelines"]["forward_two_pass_vs_concat"]["split_pass"] = BenchmarkGroup()
SUITE["pipelines"]["forward_two_pass_vs_concat"]["concat_pass"] = BenchmarkGroup()

for (label, case) in CASES
    SUITE["pipelines"]["forward_two_pass_vs_concat"]["split_pass"][label] = @benchmarkable(
        begin
            NNFoil.forward!(y, NETWORK_PARAMETERS, tmp)
            NNFoil.forward!(y_flipped, NETWORK_PARAMETERS, tmp_flipped)
        end,
        setup = (
            case = $case;
            x = copy(case.FEATURES);
            x_flipped = copy(case.FEATURES);
            NNFoil.flip_inputs!(x_flipped);
            (y, tmp) = NNFoil.allocate_forward_cache(NETWORK_PARAMETERS, x);
            (y_flipped, tmp_flipped) = NNFoil.allocate_forward_cache(NETWORK_PARAMETERS, x_flipped)
        ),
        evals = 1,
    )

    SUITE["pipelines"]["forward_two_pass_vs_concat"]["concat_pass"][label] = @benchmarkable(
        NNFoil.forward!(y, NETWORK_PARAMETERS, tmp),
        setup = (
            case = $case;
            x = copy(case.FEATURES);
            x_flipped = copy(case.FEATURES);
            NNFoil.flip_inputs!(x_flipped);
            x_both = hcat(x, x_flipped);
            (y, tmp) = NNFoil.allocate_forward_cache(NETWORK_PARAMETERS, x_both)
        ),
        evals = 1,
    )
end
# }}}

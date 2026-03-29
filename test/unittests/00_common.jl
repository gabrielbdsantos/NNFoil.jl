const UNIT_COORDINATES_RAW = readdlm(
    abspath(
        joinpath(
            NNFoil.DATA_PATH, splitpath("../test/airfoils/raw/naca0018.dat")...
        )
    )
)
const UNIT_COORDINATES = NNFoil.normalize_coordinates!(copy(UNIT_COORDINATES_RAW))

const UNIT_KULFAN = NNFoil.KulfanParameters(copy(UNIT_COORDINATES))
const UNIT_SYNTH_KULFAN = NNFoil.KulfanParameters(
    upper_weights = collect(0.1:0.1:0.8),
    lower_weights = collect(-0.05:-0.05:-0.4),
    leading_edge_weight = 0.03,
    trailing_edge_thickness = 0.01,
)
const UNIT_NETWORK_PARAMETERS = NNFoil.NeuralNetworkParameters(; model_size = :xsmall)

const UNIT_ALPHA = collect(-5.0:5.0)
const UNIT_REYNOLDS_SCALAR = 5.0e6
const UNIT_REYNOLDS_VECTOR = collect(range(3.0e6, 8.0e6; length = length(UNIT_ALPHA)))

unit_feature_matrix(n::Integer) = Matrix{Float64}(undef, 25, n)

function unit_output_buffer(n::Integer)
    return NNFoil.NeuralNetworkOutput(
        analysis_confidence = zeros(n),
        CL = zeros(n),
        CD = zeros(n),
        CM = zeros(n),
        Top_Xtr = zeros(n),
        Bot_Xtr = zeros(n),
    )
end

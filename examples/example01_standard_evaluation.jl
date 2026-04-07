using NNFoil
using DelimitedFiles

coordinates = readdlm(
    abspath(
        joinpath(
            NNFoil.DATA_PATH, splitpath("../test/airfoils/raw/naca0018.dat")...
        )
    )
)
kulfan_parameters = KulfanParameters(normalize_coordinates!(coordinates))
network_parameters = NeuralNetworkParameters(; model_size = :xlarge)
alpha = -180:180
Reynolds = 5.0e6

analysis = evaluate(network_parameters, kulfan_parameters, alpha, Reynolds)

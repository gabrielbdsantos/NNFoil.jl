import NNFoil
import DelimitedFiles

coordinates = DelimitedFiles.readdlm(joinpath(
    NNFoil.DATA_PATH, split("../test/airfoils/raw/naca0018.dat", "/")...))
kulfan_parameters = NNFoil.KulfanParameters(NNFoil.normalize_coordinates!(coordinates))
network_parameters = NNFoil.NeuralNetworkParameters(; model_size = :xlarge)
alpha = -180:180
Reynolds = 5e6

analysis = NNFoil.evaluate(network_parameters, kulfan_parameters, alpha, Reynolds)

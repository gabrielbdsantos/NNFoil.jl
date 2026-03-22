"""
`NNFoil.jl` -- a partial Julia translation of
[NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) v0.3.2.
"""
module NNFoil

import NPZ
import LinearAlgebra
import LsqFit

using ConcreteStructs: @concrete

const DATA_PATH = joinpath(@__DIR__, "..", "data")

include("kulfan.jl")
include("neural_network.jl")

export
    KulfanParameters, NeuralNetworkParameters, NeuralNetworkOutput,
    NeuralNetworkCache, build_features, update_features!,
    evaluate, evaluate!, normalize_coordinates!

end

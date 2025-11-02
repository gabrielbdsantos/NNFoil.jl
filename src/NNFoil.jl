"""
`NNFoil.jl` -- a partial Julia translation of NeuralFoil.
"""
module NNFoil

import NPZ
import LsqFit

const DATA_PATH = joinpath(@__DIR__, "..", "data")

include("kulfan.jl")
include("neural_network.jl")

end

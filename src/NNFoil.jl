module NNFoil

import NPZ
import LsqFit

const DATA_PATH = joinpath(@__DIR__, "..", "data")

include("types.jl")
include("kulfan.jl")
include("neural_network.jl")

end

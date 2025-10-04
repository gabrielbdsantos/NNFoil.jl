using DelimitedFiles: readdlm
using NNFoil: KulfanParameters, NeuralNetworkOutput

import Base.isapprox
import Base.-


function -(a::KulfanParameters, b::KulfanParameters)
    return KulfanParameters(
        upper_weights = a.upper_weights .- b.upper_weights,
        lower_weights = a.lower_weights .- a.lower_weights,
        leading_edge_weight = a.leading_edge_weight .- b.leading_edge_weight,
        trailing_edge_thickness = a.TE_thickness .- b.TE_thickness
    )
end


function isapprox(a::KulfanParameters, b::KulfanParameters; kwargs...) all(
        stack([
            isapprox.(a.upper_weights, b.upper_weights; kwargs...);
            isapprox.(a.lower_weights, b.lower_weights; kwargs...);
            isapprox.(a.leading_edge_weight, b.leading_edge_weight; kwargs...);
            isapprox.(a.trailing_edge_thickness, b.trailing_edge_thickness; kwargs...)
        ])
    )
end


function isapprox(a::NeuralNetworkOutput, b::NeuralNetworkOutput; kwargs...) all(
        stack([
            isapprox.(a.analysis_confidence, b.analysis_confidence; kwargs...);
            isapprox.(a.CL, b.CL; kwargs...);
            isapprox.(a.CD, b.CD; kwargs...);
            isapprox.(a.CM, b.CM; kwargs...);
            isapprox.(a.Top_Xtr, b.Top_Xtr; kwargs...);
            isapprox.(a.Bot_Xtr, b.Bot_Xtr; kwargs...);
        ])
    )
end


# NOTE: This is a simple normalization scheme used only to ensure that all x
# coordinates are scaled consistently --- i.e., that the x values lie within
# the unit interval [0, 1].
@inline function normalize_coordinates!(coords)
    coords[:, 1] .-= minimum(coords[:, 1])
    coords ./= maximum(coords[:, 1])
end


coordinates_from_file(filepath) = normalize_coordinates!(readdlm(filepath))

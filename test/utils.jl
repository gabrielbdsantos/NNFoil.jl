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


function isapprox(a::KulfanParameters, b::KulfanParameters; kwargs...)
    return all(
        stack(
            [
                isapprox.(a.upper_weights, b.upper_weights; kwargs...);
                isapprox.(a.lower_weights, b.lower_weights; kwargs...);
                isapprox.(a.leading_edge_weight, b.leading_edge_weight; kwargs...);
                isapprox.(a.trailing_edge_thickness, b.trailing_edge_thickness; kwargs...)
            ]
        )
    )
end


function isapprox(a::NeuralNetworkOutput, b::NeuralNetworkOutput; kwargs...)
    return all(
        stack(
            [
                isapprox.(a.analysis_confidence, b.analysis_confidence; kwargs...);
                isapprox.(a.CL, b.CL; kwargs...);
                isapprox.(a.CD, b.CD; kwargs...);
                isapprox.(a.CM, b.CM; kwargs...);
                isapprox.(a.Top_Xtr, b.Top_Xtr; kwargs...);
                isapprox.(a.Bot_Xtr, b.Bot_Xtr; kwargs...);
            ]
        )
    )
end


coordinates_from_file(filepath) = normalize_coordinates!(readdlm(filepath))

import Base.isapprox
import Base.-

function (-)(a::KulfanParameters, b::KulfanParameters)
    return KulfanParameters(
        upper_weights = a.upper_weights .- b.upper_weights,
        lower_weights = a.lower_weights .- a.lower_weights,
        leading_edge_weight = a.leading_edge_weight .- b.leading_edge_weight,
        trailing_edge_thickness = a.TE_thickness .- b.TE_thickness
    )
end

for T in (KulfanParameters, NeuralNetworkOutput)
    eval(
        quote
            function isapprox(a::$T, b::$T; kwargs...)
                return all(
                    all(isapprox.(getfield(a, field), getfield(b, field); kwargs...))
                        for field in fieldnames($T)
                )
            end
        end
    )
end

function coordinates_from_file(filepath)
    coordinates = readdlm(filepath)
    normalize_coordinates!(coordinates)
    return coordinates
end

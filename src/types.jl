"""
    KulfanParameters{T,V}

Parameter container for the Kulfan (CST) airfoil shape parameterization.

# Type parameters
- `T<:Real`: floating-point element type.
- `V<:AbstractVector{T}`: concrete vector type used for the weight arrays.

# Fields
- `upper_weights::V`: weights for the *upper* surface (commonly length 8).
- `lower_weights::V`: weights for the *lower* surface (commonly length 8).
- `leading_edge_weight::T`: scalar parameter controlling leading-edge thickness/rounding.
- `trailing_edge_thickness::T`: scalar trailing-edge thickness parameter.
"""
@kwdef struct KulfanParameters{T <: Real, V <: AbstractVector{T}}
    upper_weights::V
    lower_weights::V
    leading_edge_weight::T
    trailing_edge_thickness::T
end

"""
    KulfanParameters(upper_weights, lower_weights, leading_edge_weight,
        trailing_edge_thickness)

Alternative constructor for `KulfanParameters` that **promotes** all inputs to a
common floating-point type `T` and returns a `KulfanParameters{T, V}` where `V`
matches the concrete vector type of the provided weights arrays (after promotion).

# Arguments
- `upper_weights::AbstractVector{<:Real}`: upper weights.
- `lower_weights::AbstractVector{<:Real}`: lower weights.
- `leading_edge_weight::Real`: LE scalar parameter.
- `trailing_edge_thickness::Real`: TE thickness parameter.
"""
function KulfanParameters(
        upper_weights::Vu,
        lower_weights::Vl,
        leading_edge_weight::Tl,
        trailing_edge_thickness::Tt
) where {
        Vu <: AbstractVector{<:Real},
        Vl <: AbstractVector{<:Real},
        Tl <: Real,
        Tt <: Real
}
    T = promote_type(eltype(Vu), eltype(Vl), Tt, Tl)

    KulfanParameters(
        convert.(T, upper_weights),
        convert.(T, lower_weights),
        convert(T, leading_edge_weight),
        convert(T, trailing_edge_thickness)
    )
end

"""
    NeuralNetworkParameters{R, V, M, W, B}

Stores the parameters of the pretrained neural network model.

# Type Parameters
- `R<:Real`: numeric type used for all elements.
- `V<:AbstractVector{R}`: vector type (for input means and biases).
- `M<:AbstractMatrix{R}`: matrix type (for covariances and weight matrices).
- `W<:AbstractVector{M}`: collection of weight matrices, one per layer.
- `B<:AbstractVector{V}`: collection of bias vectors, one per layer.

# Fields
- `mean_inputs_scaled::V`: mean values of the scaled input features.
- `cov_inputs_scaled::M`: covariance matrix of the scaled inputs.
- `inv_cov_inputs_scaled::M`: inverse of the covariance matrix.
- `weights::W`: vector of weight matrices for each layer.
- `biases::B`: vector of bias vectors for each layer.
"""
struct NeuralNetworkParameters{
    R <: Real,
    V <: AbstractVector{R},
    M <: AbstractMatrix{R},
    W <: AbstractVector{M},
    B <: AbstractVector{V}
}
    mean_inputs_scaled::V
    cov_inputs_scaled::M
    inv_cov_inputs_scaled::M
    weights::W
    biases::B
end

"""
    NeuralNetworkParameters(; model_size=:xlarge, T=Float64)

Alternative constructor that loads and converts the pretrained neural network
parameters.

# Arguments
- `model_size`: Size of the pretrained model parameters to load.
- `T::Type`: Numerical type to which all loaded arrays will be
    converted.

# Returns
- `NeuralNetworkParameters`
"""
function NeuralNetworkParameters(; model_size=:xlarge, T=Float64)
    scaled_input_distribution = NPZ.npzread(
        joinpath(DATA_PATH, "scaled_input_distribution.npz")
    )
    network_parameters = NPZ.npzread(
        joinpath(DATA_PATH, "nn-" * string(model_size) * ".npz")
    )

    return NeuralNetworkParameters(
        convert.(T, scaled_input_distribution["mean_inputs_scaled"]),
        convert.(T, scaled_input_distribution["cov_inputs_scaled"]),
        convert.(T, scaled_input_distribution["inv_cov_inputs_scaled"]),
        [convert.(T, network_parameters["net.$(id).weight"])
         for id in 0:2:(length(network_parameters) - 2)],
        [convert.(T, network_parameters["net.$(id).bias"])
         for id in 0:2:(length(network_parameters) - 2)]
    )
end

"""
    NeuralNetworkOutput{V}

Stores the aerodynamic coefficients predicted by the neural network.

# Type Parameters
- `V<:AbstractVector{<:Real}`: numeric vector type used for all output quantities.

# Fields
- `analysis_confidence::V`: confidence level of the neural network prediction.
- `CL::V`: lift coefficient values.
- `CD::V`: drag coefficient values.
- `CM::V`: moment coefficient values.
- `Top_Xtr::V`: transition location on the upper surface.
- `Bot_Xtr::V`: transition location on the lower surface.
"""
@kwdef struct NeuralNetworkOutput{V <: AbstractVector{<:Real}}
    analysis_confidence::V
    CL::V
    CD::V
    CM::V
    Top_Xtr::V
    Bot_Xtr::V
end

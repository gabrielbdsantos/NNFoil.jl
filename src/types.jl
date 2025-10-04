"""
    KulfanParameters{T,V}

Parameter container for the *Kulfan* (CST) airfoil shape formulation.

This type stores the upper and lower Bernstein polynomial weights, along with
leading-edge and trailing-edge scalar parameters, all sharing a common
floating-point element type `T`. The coordinate arrays are stored with a
concrete vector type `V<:AbstractVector{T}` (e.g. `Vector{T}`, `SVector{N,T}`).

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

Converting constructor for `KulfanParameters` that **promotes** all inputs to a
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

Stores all parameters of the pretrained neural network model.

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
@kwdef struct NeuralNetworkParameters{
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
    NeuralNetworkOutput{V}

Stores the outputs predicted by the neural network.

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

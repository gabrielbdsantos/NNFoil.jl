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
        Tt <: Real,
}
    T = promote_type(eltype(Vu), eltype(Vl), Tt, Tl)

    KulfanParameters(
        convert.(T, upper_weights),
        convert.(T, lower_weights),
        convert(T, leading_edge_weight),
        convert(T, trailing_edge_thickness)
    )
end

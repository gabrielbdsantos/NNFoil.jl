"""
    KulfanParameters{Vu, Vl, Tl, Tt}

Parameter container for the Kulfan (CST) airfoil shape parameterization.

# Fields

- `upper_weights::Vu`: Weights for the *upper* surface.
- `lower_weights::Vl`: Weights for the *lower* surface.
- `leading_edge_weight::Tl`: Scalar parameter controlling leading-edge
  thickness/rounding.
- `trailing_edge_thickness::Tt`: Scalar trailing-edge thickness parameter.
"""
@kwdef struct KulfanParameters{Vu, Vl, Tl, Tt}
    upper_weights::Vu
    lower_weights::Vl
    leading_edge_weight::Tl
    trailing_edge_thickness::Tt
end

"""
    KulfanParameters(coordinates)

Fits Kulfan (CST) parameters to a set of airfoil coordinates.

This method assumes that the input coordinates follow the **Selig ordering**,
i.e., starting at the trailing edge, proceeding along the upper surface to the
leading edge, and returning along the lower surface. The number of Bernstein
weights per surface is currently fixed internally at eight, following the
implementation used in [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil).

# Arguments

- `coordinates::AbstractMatrix`: Airfoil coordinates with columns `[x, y]`.

# Returns

- [`KulfanParameters`](@ref)
"""
function KulfanParameters(coordinates)
    weights_per_side = 8
    coords_upper, coords_lower = split_upper_lower_surfaces(coordinates)

    x_upper = @view coords_upper[:, 1]
    y_upper = @view coords_upper[:, 2]
    x_lower = @view coords_lower[:, 1]
    y_lower = @view coords_lower[:, 2]

    trailing_edge_thickness = y_upper[1] - y_lower[end]

    fit = LsqFit.curve_fit(
        (x, p) -> cst_y_coordinates(x, p, length(x_upper)),
        [x_upper; x_lower],
        [y_upper; y_lower],
        [ones(2 * weights_per_side + 1); trailing_edge_thickness],
        autodiff = :forwarddiff
    )

    offset = 1

    if fit.param[end] < 0
        offset = 0

        fit = LsqFit.curve_fit(
            (x, p) -> cst_y_coordinates(x, [p; 0], length(x_upper)),
            [x_upper; x_lower],
            [y_upper; y_lower],
            ones(2 * weights_per_side + 1),
            autodiff = :forwarddiff
        )
    end

    return KulfanParameters(
        upper_weights = fit.param[1:weights_per_side],
        lower_weights = fit.param[(weights_per_side + 1):(2 * weights_per_side)],
        leading_edge_weight = fit.param[end - offset],
        trailing_edge_thickness = offset * fit.param[end],
    )
end

"""
    normalize_coordinates!(coordinates)

Normalize the input coordinates in place so that the x values lie within the
unit interval [0, 1].

# Arguments

- `coordinates::AbstractMatrix`: Matrix of airfoil coordinates with columns
  representing the x and y values.

!!! warning
    The current normalization is a temporary workaround and may be revised in
    future versions of NNFoil so that it more closely matches how NeuralFoil
    normalizes coordinates.
"""
function normalize_coordinates!(coordinates)
    coordinates[:, 1] .-= minimum(@view coordinates[:, 1])
    return coordinates ./= maximum(@view coordinates[:, 1])
end

"""
    split_upper_lower_surfaces(coordinates)

Split airfoil coordinates into upper and lower surfaces.

# Arguments

- `coordinates::AbstractMatrix`: Matrix of airfoil coordinates with columns
  representing the x and y values.

# Returns

- `(upper, lower)`: Two matrices containing the coordinates of the upper and
  lower surfaces, respectively.
"""
@inline function split_upper_lower_surfaces(coordinates)
    _, n = findmin(@view coordinates[:, 1])
    offset = isodd(size(coordinates, 1)) ? 0 : 1

    return coordinates[1:n, :], coordinates[(n + offset):end, :]
end

"""
    bernstein(x, v, n)

Evaluate the Bernstein basis polynomial of degree `n` and index `v` at `x`.

# Arguments

- `x`: Evaluation points (scalar, vector, or array).
- `v`: Index of the basis polynomial (`0 ≤ v ≤ n`).
- `n`: Degree of the polynomial.

# Returns

- Array of the same shape as `x`: Values of the Bernstein polynomial.
"""
@inline function bernstein(x, v, n)
    return @. binomial(n, v) * x^v * (1 - x)^(n - v)
end

"""
    class_function(x)

Evaluate the class function used in Kulfan’s parametrization.

In NeuralFoil, the class-shape exponents N1 and N2 are hardcoded as N1=0.5 and
N2=1.0.

# Arguments

- `x`: Nondimensional chordwise coordinates [0--1].

# Returns

- Array of the same shape as `x`: Values of the class function.
"""
@inline function class_function(x)
    return @. sqrt(x) * (1 - x)
end

"""
    shape_function(x, coefficients)

Kulfan shape function defined as a weighted sum of Bernstein polynomials.

# Arguments

- `x`: Nondimensional chordwise coordinates.
- `coefficients::AbstractVector`: Weights for the Bernstein polynomials.

# Returns

- Same shape as `x`: Values of the shape function.
"""
@inline function shape_function(x, coefficients)
    S = similar(x) .= 0

    for (i, c) in enumerate(coefficients)
        S += c * bernstein(x, i - 1, length(coefficients) - 1)
    end

    return S
end

"""
    cst(x, coefficients, leading_edge_weight, trailing_edge_thickness)

CST (Class--Shape Transformation) surface parametrization.

# Arguments

- `x`: Nondimensional chordwise coordinates [0, 1].
- `coefficients::AbstractVector`: Shape function weights.
- `leading_edge_weight::Real`: Leading-edge modification term.
- `trailing_edge_thickness::Real`: Trailing-edge thickness parameter.

# Returns

- Same shape as `x`: Airfoil surface coordinates defined by the CST
  parametrization.
"""
function cst(x, coefficients, leading_edge_weight, trailing_edge_thickness)
    N = length(coefficients)
    C = class_function(x)
    S = shape_function(x, coefficients)

    return @. C * S +
        x * trailing_edge_thickness +
        leading_edge_weight * x * max(1 - x, 0)^(N + 0.5)
end

"""
    cst_y_coordinates(x, parameters, x_split_id)

Generate the airfoil surface coordinates from Kulfan parameters.

# Arguments

- `x::AbstractVector`: Nondimensional chordwise coordinates, following the
  Selig ordering.
- `parameters::AbstractVector`: Vector containing the CST parameters: upper and
  lower weights, leading-edge weight, and trailing-edge thickness.
- `x_split_id::Int`: Index separating the upper and lower surface coordinates
  in `x`.

# Returns

- `Vector{<:Real}`: Airfoil surface y-coordinates corresponding to `x`.
"""
function cst_y_coordinates(x, parameters, x_split_id)
    weights..., leading_edge_weight, trailing_edge_gap = parameters

    N = convert(Int, length(weights) / 2)
    weights_upper = weights[1:N]
    weights_lower = weights[(N + 1):end]

    x_upper = x[1:x_split_id]
    x_lower = x[(x_split_id + 1):end]

    y_upper = cst(
        reverse(x_upper), weights_upper, leading_edge_weight, trailing_edge_gap / 2
    )
    y_lower = cst(
        x_lower, weights_lower, leading_edge_weight, -trailing_edge_gap / 2
    )

    return [reverse(y_upper); y_lower]
end

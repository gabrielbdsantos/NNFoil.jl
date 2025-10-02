"""
    normalize_coordinates!(coordinates)

Normalize the input coordinates in place so that the x values lie within the unit interval [0, 1].

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
    coordinates ./= maximum(@view coordinates[:, 1])
end


"""
    split_upper_lower_surfaces(coordinates)

Split airfoil coordinates into upper and lower surfaces.

# Arguments
- `coordinates::AbstractMatrix`: Matrix of airfoil coordinates with columns
  representing the x and y values.

# Returns
- `(upper, lower)`: Two matrices containing the coordinates of the upper
  and lower surfaces, respectively.
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
- `v::Signed`: Index of the basis polynomial.
- `n::Signed`: Degree of the polynomial.

# Returns
- Array of the same shape as `x`: Values of the Bernstein polynomial.
"""
@inline function bernstein(x, v::I, n::I) where {I <: Signed}
    return @. binomial(n, v) * x^v * (1 - x)^(n - v)
end


"""
    class_function(x, N1, N2)

Evaluate the class function used in Kulfan’s parametrization.

# Arguments
- `x`: Nondimensional chordwise coordinates [0--1].
- `N1::Real`: Leading-edge exponent.
- `N2::Real`: Trailing-edge exponent.

# Returns
- Array of the same shape as `x`: Values of the class function
"""
@inline function class_function(x, N1, N2)
    return @. (x^N1) * (1 - x)^N2
end


"""
    shape_function(x, coefficients)

Kulfan shape function defined as a weighted sum of Bernstein polynomials.

# Arguments
- `x`: Nondimensional chordwise coordinates [0, 1].
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
    cst(x, coefficients, leading_edge_weight, trailing_edge_thickness; N1=0.5, N2=1.0)

CST (Class--Shape Transformation) airfoil parametrization.

# Arguments
- `x`: Nondimensional chordwise coordinates [0, 1]
- `coefficients::AbstractVector`: Shape function weights
- `leading_edge_weight::Real`: Leading-edge modification term
- `trailing_edge_thickness::Real`: Trailing-edge thickness parameter
- `N1::Real`: Leading-edge exponent (default: 0.5)
- `N2::Real`: Trailing-edge exponent (default: 1.0)

# Returns
- Same shape as `x`: Airfoil surface coordinates defined by the CST parametrization
"""
function cst(
        x, coefficients, leading_edge_weight, trailing_edge_thickness;
        N1 = 0.5, N2 = 1.0
)
    N = length(coefficients)
    C = class_function(x, N1, N2)
    S = shape_function(x, coefficients)

    return @. C * S +
              x * trailing_edge_thickness +
              leading_edge_weight * x * max(1 - x, 0)^(N + 0.5)
end


"""
    airfoil_cst(x, parameters, x_split_id; N1=0.5, N2=1.0)

Reconstruct an airfoil surface from Kulfan (CST) parameters.

# Arguments
- `x`: Vector of nondimensional chordwise coordinates (0–1)
- `parameters::AbstractVector`: upper and lower weights, leading-edge weight,
    trailing-edge thickness
- `x_split_id::Int`: Index separating upper and lower surface coordinates
- `N1::Real`: Leading-edge exponent (default: 0.5)
- `N2::Real`: Trailing-edge exponent (default: 1.0)

# Returns
- `Vector`: Airfoil surface y-coordinates corresponding to `x`
"""
function airfoil_cst(x, parameters, x_split_id; N1 = 0.5, N2 = 1.0)
    weights..., leading_edge_weight, trailing_edge_gap = parameters

    N = convert(Int, length(weights) / 2)
    weights_upper = weights[1:N]
    weights_lower = weights[(N + 1):end]

    x_upper = x[1:x_split_id]
    x_lower = x[(x_split_id + 1):end]

    y_upper = cst(
        reverse(x_upper), weights_upper, leading_edge_weight, trailing_edge_gap / 2; N1, N2
    )
    y_lower = cst(
        x_lower, weights_lower, leading_edge_weight, -trailing_edge_gap / 2; N1, N2
    )

    return [reverse(y_upper); y_lower]
end

"""
    airfoil_cst_zero_trailing_edge(x, parameters, x_split_id; N1=0.5, N2=1.0)

Reconstruct an airfoil surface from Kulfan (CST) parameters with a zero trailing-edge gap.

# Arguments
- `x`: Vector of nondimensional chordwise coordinates (0–1)
- `parameters::AbstractVector`: upper and lower weights, and leading-edge weight
- `x_split_id::Int`: Index separating upper and lower surface coordinates
- `N1::Real`: Leading-edge exponent (default: 0.5)
- `N2::Real`: Trailing-edge exponent (default: 1.0)

# Returns
- `Vector`: Airfoil surface y-coordinates corresponding to `x`
"""
function airfoil_cst_zero_trailing_edge(x, parameters, x_split_id; N1 = 0.5, N2 = 1.0)
    return airfoil_cst(x, [parameters; 0], x_split_id; N1, N2)
end


"""
    get_kulfan_parameters(coordinates; num_coefficients=8, N1=0.5, N2=1.0)

Fit Kulfan (CST) parameters to airfoil coordinates.

# Arguments
- `coordinates::AbstractMatrix`: Airfoil coordinates with columns `[x, y]`
- `num_coefficients::Int`: Number of Bernstein polynomial coefficients per surface
    (default: 8)
- `N1::Real`: Leading-edge exponent (default: 0.5)
- `N2::Real`: Trailing-edge exponent (default: 1.0)

# Returns
- `KulfanParameters`: Fitted upper and lower weights, leading-edge weight, and
    trailing-edge thickness
"""
function get_kulfan_parameters(coordinates; num_coefficients = 8, N1 = 0.5, N2 = 1.0)
    coords_upper, coords_lower = split_upper_lower_surfaces(coordinates)

    x_upper = @view coords_upper[:, 1]
    y_upper = @view coords_upper[:, 2]
    x_lower = @view coords_lower[:, 1]
    y_lower = @view coords_lower[:, 2]

    trailing_edge_thickness = y_upper[1] - y_lower[end]

    fit = LsqFit.curve_fit(
        (x, y) -> airfoil_cst(x, y, length(x_upper); N1, N2),
        [x_upper; x_lower],
        [y_upper; y_lower],
        [ones(2 * num_coefficients + 1); trailing_edge_thickness],
        autodiff=:forwarddiff
    )

    if fit.param[end] < 0
        fit = LsqFit.curve_fit(
            (x, y) -> airfoil_cst_zero_trailing_edge(x, y, length(x_upper); N1, N2),
            [x_upper; x_lower],
            [y_upper; y_lower],
            ones(2 * num_coefficients + 1),
            autodiff=:forwarddiff
        )

        upper_weights = fit.param[1:num_coefficients]
        lower_weights = fit.param[(num_coefficients + 1):(end - 1)]
        leading_edge_weight = fit.param[end]
        trailing_edge_thickness = convert(typeof(leading_edge_weight), 0)

        res = [fit.param; 0]
    else
        upper_weights = fit.param[1:num_coefficients]
        lower_weights = fit.param[(num_coefficients + 1):(end - 2)]
        leading_edge_weight = fit.param[end - 1]
        trailing_edge_thickness = fit.param[end]

        res = fit.param
    end

    return KulfanParameters(
        upper_weights = res[1:num_coefficients],
        lower_weights = res[(num_coefficients + 1):(2 * num_coefficients)],
        leading_edge_weight = res[end - 1],
        trailing_edge_thickness = res[end],
    )
end

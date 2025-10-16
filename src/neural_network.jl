"""
    load_network_parameters(; model_size=:xlarge, T=Float64)

Load and convert the pretrained neural network parameters used by
NeuralFoil.

# Arguments
- `model_size::Symbol=:xlarge`: Size of the pretrained model parameters to load.
- `T::Type=Float64`: Numerical type to which all loaded arrays will be
   converted.
"""
function load_network_parameters(; model_size = :xlarge, T = Float64)
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
    swish(x; beta=1)

Apply the Swish activation function elementwise.

# Arguments
- `x`: Input value.
- `beta::Real=1`: Slope parameter controlling smoothness.

# Returns
- Scalar or Array of the same shape as `x`: Activated values.
"""
@inline function swish(x; beta = 1)
    return @. x / (1 + exp(-beta * x))
end

"""
    sigmoid(x::T; ln_eps=log(10 / floatmax(eltype(T)))) where {T}

Apply the sigmoid activation function elementwise. Also employ input clipping for
numerical stability.

# Arguments
- `x::T`: Input value.
- `ln_eps::Real=log(10 / floatmax(eltype(T)))`: Logarithmic bound used to
    clip input values for stability.

# Returns
- Scalar or Array of the same shape as `x`: Values in the range (0, 1).
"""
@inline function sigmoid(x::T; ln_eps = log(10 / floatmax(eltype(T)))) where {T}
    x_clipped = clamp.(x, ln_eps, -ln_eps)
    return 1 ./ (1 .+ exp.(-x_clipped))
end

"""
    squared_mahalanobis_distance(network_parameters::NetworkParameters,
        x::AbstractMatrix{<:Real})

Compute the squared Mahalanobis distance between input samples and the mean of the
scaled input distribution.

# Arguments
- `network_parameters::NetworkParameters`: pretrained neural network parameters containing
    the mean and inverse covariance of the scaled input distribution.
- `x::AbstractArray{<:Real}`: Input samples.

# Returns
- `AbstractArray{<:Real}`
"""
function squared_mahalanobis_distance(network_parameters::NeuralNetworkParameters,
        x::T) where {T <: AbstractArray{<:Real}}
    x_minus_mean = (x .- network_parameters.mean_inputs_scaled)'

    return sum(
        (x_minus_mean * network_parameters.inv_cov_inputs_scaled) .* x_minus_mean; dims = 2)
end

"""
    net(network_parameters::NetworkParameters, x::AbstractMatrix{<:Real})

Evaluate the neural network using the pretrained network parameters on the given input `x`.

# Arguments
- `network_parameters::NetworkParameters`: pretrained network weights and biases.
- `x::AbstractArray{<:Real}`: Input data.

# Returns
- `AbstractMatrix{<:Real}`
"""
function net(network_parameters::NeuralNetworkParameters, x::AbstractArray{<:Real})
    weights = network_parameters.weights
    biases = network_parameters.biases

    for (i, (W, b)) in enumerate(zip(weights, biases))
        x = W * x .+ b

        if i != length(weights)
            x = swish(x)
        end
    end

    return x
end

"""
    __flip_x!(x)

Flip the input vector or matrix `x` in-place, creating a geometrically mirrored
version of the input features.

# Arguments
- `x::AbstractVecOrMat{<:Real}`: Input vector or matrix where each column
    represents a sample. The flipping is applied across specific rows.

# Returns
- `x`: The modified input matrix, flipped in-place.
"""
function __flip_x!(x::T) where {T <: AbstractVecOrMat{<:Real}}
    @inbounds for i in axes(x, 2)
        for j in 1:8
            x[j, i], x[(8 + j), i] = -x[(8 + j), i], -x[j, i]
        end

        x[17, i] *= -1
        x[19, i] *= -1
        x[(end - 1), i], x[end, i] = x[end, i], x[(end - 1), i]
    end

    return x
end

"""
    __evaluate_aero(network_parameters, x) -> NeuralNetworkOutput

Evaluate the neural network for an input `x` using the pretrained parameters
`network_parameters`, performing symmetry fusion and applying post-processing
transformations to produce physically meaningful aerodynamic coefficients.

# Arguments
- `network_parameters`: Pretrained parameters of the neural network.
- `x::AbstractMatrix`: Input features for the original flow condition. Each
    column corresponds to one input sample.

# Returns
A `NeuralNetworkOutput` struct with fields:
- `analysis_confidence`: confidence level (0--1)
- `CL`: lift coefficient
- `CD`: drag coefficient
- `CM`: moment coefficient
- `Top_Xtr`: upper-surface transition location (0--1)
- `Bot_Xtr`: lower-surface transition location (0--1)

!!! note
    Currently, boundary-layer related outputs are not supported. These are
    planned to be included in a future version.
"""
function __evaluate_aero(network_parameters, x)
    y = net(network_parameters, x)
    y[1, :] .-= (
        squared_mahalanobis_distance(network_parameters, x)
        ./ (2.0 * size(x, 1))
    )

    x_flipped = __flip_x!(copy(x))

    y_flipped = net(network_parameters, x_flipped)
    y_flipped[1, :] .-= (
        squared_mahalanobis_distance(network_parameters, x_flipped)
        ./
        (2.0 * size(x_flipped, 1))
    )

    # Temporary variable
    tmp = y_flipped[6, :]

    y_flipped[2, :] .= -y_flipped[2, :]
    y_flipped[4, :] .= -y_flipped[4, :]
    y_flipped[6, :] .= y_flipped[5, :]
    y_flipped[5, :] .= tmp

    # Average outputs
    y_fused = @. (y + y_flipped) / 2.0

    # Analysis confidence
    @. y_fused[1, :] = sigmoid(y_fused[1, :])

    # Lift coefficient
    @. y_fused[2, :] = y_fused[2, :] / 2.0

    # Drag coefficient
    @. y_fused[3, :] = exp((y_fused[3, :] - 2) * 2)

    # Moment coefficient
    @. y_fused[4, :] = y_fused[4, :] / 20

    # Top and bottom transitions
    @. y_fused[5, :] = clamp(y_fused[5, :], 0, 1)
    @. y_fused[6, :] = clamp(y_fused[6, :], 0, 1)

    # TODO: support boundary layer outputs
    # ...

    return NeuralNetworkOutput(
        analysis_confidence = y_fused[1, :],
        CL = y_fused[2, :],
        CD = y_fused[3, :],
        CM = y_fused[4, :],
        Top_Xtr = y_fused[5, :],
        Bot_Xtr = y_fused[6, :]
    )
end

"""
    get_aero_from_kulfan_parameters(
        network_parameters,
        kulfan_parameters,
        alpha,
        Reynolds;
        n_crit=9,
        xtr_upper=1,
        xtr_lower=1
    ) -> NeuralNetworkOutput

Compute aerodynamic coefficients from Kulfan airfoil parameters using a
pretrained neural network.

This function supports scalar and vector inputs for both the angle of attack
`alpha` and the `Reynolds` number. If either is a scalar while the other is a
vector, the scalar will be broadcasted to match the length of the vector.

# Arguments
- `network_parameters::NeuralNetworkParameters`: Pretrained neural network parameters.
- `kulfan_parameters::KulfanParameters`: Kulfan shape parameters describing the airfoil.
- `alpha`: Angle(s) of attack in degrees (`Real` or `AbstractVector{<:Real}`).
- `Reynolds`: Reynolds number(s) corresponding to each `alpha` (`Real` or `AbstractVector{<:Real}`).
- `n_crit::Real=9`: Critical amplification factor for transition prediction.
- `xtr_upper::Real=1`: Forced transition location on the upper surface.
- `xtr_lower::Real=1`: Forced transition location on the lower surface.

# Returns
- `NeuralNetworkOutput`: Predicted aerodynamic coefficients.
"""
function get_aero_from_kulfan_parameters(
        network_parameters::NeuralNetworkParameters,
        kulfan_parameters::KulfanParameters,
        alpha::Ta,
        Reynolds::Tb,
        ;
        n_crit = 9,
        xtr_upper = 1,
        xtr_lower = 1
) where {Ta <: AbstractVector{<:Real}, Tb <: AbstractVector{<:Real}}
    L = length(alpha)
    x = vcat(
        repeat(kulfan_parameters.upper_weights, outer = (1, L)),
        repeat(kulfan_parameters.lower_weights, outer = (1, L)),
        fill(kulfan_parameters.leading_edge_weight, (1, L)),
        fill(kulfan_parameters.trailing_edge_thickness * 50, (1, L)),
        sind.(2 .* alpha'),
        cosd.(alpha'),
        1 .- cosd.(alpha') .^ 2,
        (log.(Reynolds') .- 12.5) ./ 3.5,
        fill((n_crit - 9) / 4.5, (1, L)),
        fill(xtr_upper, (1, L)),
        fill(xtr_lower, (1, L))
    )

    return __evaluate_aero(network_parameters, x)
end

function get_aero_from_kulfan_parameters(
        network_parameters::NeuralNetworkParameters,
        kulfan_parameters::KulfanParameters,
        alpha::Ta,
        Reynolds::Tr,
        ;
        n_crit = 9,
        xtr_upper = 1,
        xtr_lower = 1
) where {Ta <: Real, Tr <: Real}
    x = [kulfan_parameters.upper_weights
         kulfan_parameters.lower_weights
         kulfan_parameters.leading_edge_weight
         kulfan_parameters.trailing_edge_thickness * 50.0
         sind(2.0 * alpha)
         cosd(alpha)
         1.0 - cosd(alpha)^2
         (log(Reynolds) - 12.5) / 3.5
         (n_crit - 9.0) / 4.5
         xtr_upper
         xtr_lower]

    return __evaluate_aero(network_parameters, x)
end

function get_aero_from_kulfan_parameters(
        network_parameters::NeuralNetworkParameters,
        kulfan_parameters::KulfanParameters,
        alpha::T,
        Reynolds::R,
        ;
        n_crit = 9,
        xtr_upper = 1,
        xtr_lower = 1
) where {T <: AbstractVector{<:Real}, R <: Real}
    return get_aero_from_kulfan_parameters(
        network_parameters, kulfan_parameters, alpha, fill(Reynolds, length(alpha));
        n_crit, xtr_upper, xtr_lower
    )
end

function get_aero_from_kulfan_parameters(
        network_parameters::NeuralNetworkParameters,
        kulfan_parameters::KulfanParameters,
        alpha::T,
        Reynolds::R,
        ;
        n_crit = 9,
        xtr_upper = 1,
        xtr_lower = 1
) where {T <: Real, R <: AbstractVector{<:Real}}
    return get_aero_from_kulfan_parameters(
        network_parameters, kulfan_parameters, fill(alpha, length(Reynolds)), Reynolds;
        n_crit, xtr_upper, xtr_lower
    )
end

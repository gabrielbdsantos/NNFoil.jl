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
         for id in 0:2:(length(network_parameters) - 2)],
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
@inline function swish(x; beta=1)
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
@inline function sigmoid(x::T; ln_eps=log(10 / floatmax(eltype(T)))) where {T}
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
- `x::AbstractMatrix{<:Real}`: Input samples.

# Returns
- `AbstractMatrix{<:Real}`
"""
function squared_mahalanobis_distance(
        network_parameters::NeuralNetworkParameters, x::T) where {T <: AbstractMatrix{<:Real}}
    x_minus_mean = (x .- network_parameters.mean_inputs_scaled)'

    return sum(
        (x_minus_mean * network_parameters.inv_cov_inputs_scaled) .* x_minus_mean;
        dims=2
    )
end

"""
    net(network_parameters::NetworkParameters, x::AbstractMatrix{<:Real})

Evaluate the neural network using the pretrained network parameters on the given input `x`.

# Arguments
- `network_parameters::NetworkParameters`: pretrained network weights and biases.
- `x::AbstractMatrix{<:Real}`: Input data.

# Returns
- `AbstractMatrix{<:Real}`
"""
function net(network_parameters::NeuralNetworkParameters, x::AbstractMatrix{<:Real})
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
    get_aero_from_kulfan_parameters(
        network_parameters::NetworkParameters,
        kulfan_parameters::KulfanParameters,
        alpha::AbstractVector{<:Real},
        Reynolds::AbstractVector{<:Real};
        n_crit=9,
        xtr_upper=1,
        xtr_lower=1
    )

Compute aerodynamic coefficients from Kulfan parameters using a pretrained neural
network model.

# Arguments
- `network_parameters::NetworkParameters`: Pretrained neural network parameters.
- `kulfan_parameters::KulfanParameters`: Kulfan shape parameters describing the airfoil.
- `alpha::AbstractVector{<:Real}`: Angles of attack in degrees.
- `Reynolds::AbstractVector{<:Real}`: Reynolds numbers corresponding to each `alpha`.
- `n_crit::Real=9`: Critical amplification factor for transition prediction.
- `xtr_upper::Real=1`: Forced transition location on the upper surface.
- `xtr_lower::Real=1`: Forced transition location on the lower surface.

# Returns
- `NeuralNetworkOutput`: Predicted aerodynamic coefficients.
"""
function get_aero_from_kulfan_parameters(
        network_parameters::NeuralNetworkParameters,
        kulfan_parameters::KulfanParameters,
        alpha::T,
        Reynolds::T,
        ;
        n_crit = 9,
        xtr_upper = 1,
        xtr_lower = 1,
) where {
        T <: AbstractVector{<:Real},
}
    @assert size(alpha) == size(Reynolds)

    x = stack([[kulfan_parameters.upper_weights
                kulfan_parameters.lower_weights
                kulfan_parameters.leading_edge_weight
                kulfan_parameters.trailing_edge_thickness * 50.0
                sind(2.0 * angle)
                cosd(angle)
                1.0 - cosd(angle)^2
                (log(reynolds) - 12.5) / 3.5
                (n_crit .- 9.0) / 4.5
                xtr_upper
                xtr_lower] for (angle, reynolds) in zip(alpha, Reynolds)])

    x_flipped = stack([[-kulfan_parameters.lower_weights
                        -kulfan_parameters.upper_weights
                        -kulfan_parameters.leading_edge_weight
                        kulfan_parameters.trailing_edge_thickness * 50.0
                        -sind(2.0 * angle)
                        cosd(angle)
                        1.0 - cosd(angle)^2
                        (log(reynolds) - 12.5) / 3.5
                        (n_crit .- 9.0) / 4.5
                        xtr_lower
                        xtr_upper] for (angle, reynolds) in zip(alpha, Reynolds)])


    y = net(network_parameters, x)
    y_flipped = net(network_parameters, x_flipped)

    y[1, :] .-= (
        squared_mahalanobis_distance(network_parameters, x)
        ./ (2.0 * size(x, 1))
    )
    y_flipped[1, :] .-= (
        squared_mahalanobis_distance(network_parameters, x_flipped)
        ./ (2.0 * size(x, 1))
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


function get_aero_from_kulfan_parameters(
        network_parameters::NeuralNetworkParameters,
        kulfan_parameters::KulfanParameters,
        alpha::T,
        Reynolds::T,
        ;
        n_crit = 9,
        xtr_upper = 1,
        xtr_lower = 1,
) where {T <: Real}
    get_aero_from_kulfan_parameters(
        network_parameters, kulfan_parameters, [alpha], [Reynolds];
        n_crit, xtr_upper, xtr_lower)
end


"""
    get_aero_from_kulfan_parameters(
        kulfan_parameters::KulfanParameters,
        alpha::AbstractVector{<:Real},
        Reynolds::AbstractVector{<:Real};
        n_crit=9,
        xtr_upper=1,
        xtr_lower=1,
        model_size=:xlarge
    )

Compute aerodynamic coefficients from Kulfan parameters using a pretrained neural
network model.

# Arguments
- `kulfan_parameters::KulfanParameters`: Kulfan shape parameters describing the airfoil.
- `alpha::AbstractVector{<:Real}`: Angles of attack in degrees.
- `Reynolds::AbstractVector{<:Real}`: Reynolds numbers corresponding to each `alpha`.
- `n_crit::Real=9`: Critical amplification factor for transition prediction.
- `xtr_upper::Real=1`: Forced transition location on the upper surface.
- `xtr_lower::Real=1`: Forced transition location on the lower surface.
- `model_size::Symbol=:xlarge`: Size of the pretrained model parameters to load.

# Returns
- `NeuralNetworkOutput`: Predicted aerodynamic coefficients.
"""
function get_aero_from_kulfan_parameters(
        kulfan_parameters::KulfanParameters,
        alpha::T,
        Reynolds::T,
        ;
        n_crit = 9,
        xtr_upper = 1,
        xtr_lower = 1,
        model_size = :xlarge
) where {
        T <: AbstractVector{<:Real},
}
    network_parameters = load_network_parameters(; model_size = model_size)

    return get_aero_from_kulfan_parameters(
        network_parameters, kulfan_parameters, alpha, Reynolds;
        n_crit = n_crit, xtr_upper = xtr_upper, xtr_lower = xtr_lower)
end

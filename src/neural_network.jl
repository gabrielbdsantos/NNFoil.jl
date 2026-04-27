"""
Currently supported output channels. See [`NeuralNetworkOutput`](@ref).
"""
const SUPPORTED_OUTPUT_CHANNELS = 6

# Types {{{
# ----------------------------------------------------------------------
"""
    NeuralNetworkParameters{R, V, M, W, B}

Stores the parameters of the pretrained neural network model.

# Type Parameters

- `R<:Real`: numeric type used for all elements.
- `V<:AbstractVector{R}`
- `M<:AbstractMatrix{R}`
- `W<:AbstractVector{M}`
- `B<:AbstractVector{V}`

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
        B <: AbstractVector{V},
    }
    mean_inputs_scaled::V
    cov_inputs_scaled::M
    inv_cov_inputs_scaled::M
    weights::W
    biases::B
end

"""
    NeuralNetworkParameters(; model_size=:xlarge, T=Float64)

Convenience constructor that loads and converts the pretrained neural network
parameters.

# Keyword arguments

- `model_size::Symbol`: Size of the pretrained model parameters to load.
- `T::Type`: Numerical type to which all loaded arrays will be converted.

# Returns

- [`NeuralNetworkParameters`](@ref)
"""
function NeuralNetworkParameters(; model_size = :xlarge, T = Float64)
    scaled_input_distribution = NPZ.npzread(
        joinpath(DATA_PATH, "scaled_input_distribution.npz")
    )
    network_parameters = NPZ.npzread(
        joinpath(DATA_PATH, "nn-" * string(model_size) * ".npz")
    )

    weights = [
        convert.(T, network_parameters["net.$(id).weight"])
            for id in 0:2:(length(network_parameters) - 2)
    ]
    biases = [
        convert.(T, network_parameters["net.$(id).bias"])
            for id in 0:2:(length(network_parameters) - 2)
    ]

    weights[end] = weights[end][1:SUPPORTED_OUTPUT_CHANNELS, :]
    biases[end] = biases[end][1:SUPPORTED_OUTPUT_CHANNELS]

    return NeuralNetworkParameters(
        convert.(T, scaled_input_distribution["mean_inputs_scaled"]),
        convert.(T, scaled_input_distribution["cov_inputs_scaled"]),
        convert.(T, scaled_input_distribution["inv_cov_inputs_scaled"]),
        weights,
        biases
    )
end

"""
    NeuralNetworkOutput{V}

Stores the aerodynamic coefficients predicted by the neural network.

# Type Parameters

- `T <: Real`
- `C <: AbstractVector{Union{T, AbstractVector{T}}}`

# Fields

- `analysis_confidence::C`: confidence level of the neural network prediction.
- `CL::C`: lift coefficient values.
- `CD::C`: drag coefficient values.
- `CM::C`: moment coefficient values.
- `Top_Xtr::C`: transition location on the upper surface.
- `Bot_Xtr::C`: transition location on the lower surface.

!!! note

    Boundary-layer related outputs are currently **not supported**. Support for
    these outputs is planned in a future version.
"""
@kwdef struct NeuralNetworkOutput{T <: Real, C <: Union{T, AbstractVector{T}}}
    analysis_confidence::C
    CL::C
    CD::C
    CM::C
    Top_Xtr::C
    Bot_Xtr::C
end

"""
    NeuralNetworkCache

Mutable cache that stores preallocated arrays used during repeated neural
network evaluations.

# Fields

- `parameters<:NeuralNetworkParameters`: pretrained network parameters.
- `output<:NeuralNetworkOutput`: output buffers for aerodynamic coefficients.
- `y`: network outputs for the original inputs.
- `y_flipped`: network outputs for symmetry-flipped inputs.
- `x`: original input features.
- `x_flipped`: symmetry-flipped input features.
- `tmp_x`: layer-activation cache used when evaluating `x`.
- `tmp_x_flipped`: layer-activation cache used when evaluating `x_flipped`.
- `tmp_x_smd1`: temporary workspace for Mahalanobis-distance intermediates.
- `tmp_x_smd2`: temporary workspace for Mahalanobis-distance intermediates.
- `tmp_y_smd`: temporary workspace for Mahalanobis-distance outputs.
"""
@concrete struct NeuralNetworkCache
    network_parameters <: NeuralNetworkParameters
    outputs <: NeuralNetworkOutput
    # -------------
    y
    y_flipped
    y_both
    # -------------
    x
    x_flipped
    x_both
    # -------------
    tmp_x
    tmp_x_flipped
    tmp_x_both
    # -------------
    tmp_x_smd1
    tmp_x_smd2
    tmp_y_smd
end

function NeuralNetworkCache(params::NeuralNetworkParameters, x0::AbstractMatrix)
    (L, C) = (size(x0, 1), size(x0, 2))
    x_both = similar(x0, L, 2C)

    x = @view x_both[:, 1:C]
    x_flipped = @view x_both[:, (C + 1):end]
    copyto!(x, x0)
    copyto!(x_flipped, x0)
    flip_inputs!(x_flipped)

    y_both, tmp_x_both = allocate_forward_cache(params, x_both)

    y = @view y_both[:, 1:C]
    y_flipped = @view y_both[:, (C + 1):end]

    tmp_x = [@view tmp_x_both[i][:, 1:C] for i in eachindex(tmp_x_both)]
    tmp_x_flipped = [@view tmp_x_both[i][:, (C + 1):(2C)] for i in eachindex(tmp_x_both)]

    tmp_x_smd1 = similar(x)
    tmp_x_smd2 = similar(x)
    tmp_y_smd = similar(x, size(x, 2), 1)

    output = NeuralNetworkOutput(
        (similar(y[1, :]) for _ in 1:fieldcount(NeuralNetworkOutput))...
    )

    return NeuralNetworkCache(
        params, output,
        y, y_flipped, y_both,
        x, x_flipped, x_both,
        tmp_x, tmp_x_flipped, tmp_x_both,
        tmp_x_smd1, tmp_x_smd2, tmp_y_smd
    )
end

function NeuralNetworkCache(params::NeuralNetworkParameters, x::AbstractVector)
    x_flipped = copy(x)
    flip_inputs!(x_flipped)

    y, tmp_x = allocate_forward_cache(params, x)
    y_flipped, tmp_x_flipped = allocate_forward_cache(params, x_flipped)

    tmp_x_smd1 = similar(x)
    tmp_x_smd2 = similar(x)
    tmp_y_smd = similar(x, size(x, 2), 1)

    output = NeuralNetworkOutput(
        (similar(y[1, :]) for _ in 1:fieldcount(NeuralNetworkOutput))...
    )

    return NeuralNetworkCache(
        params, output,
        y, y_flipped, nothing,
        x, x_flipped, nothing,
        tmp_x, tmp_x_flipped, nothing,
        tmp_x_smd1, tmp_x_smd2, tmp_y_smd,
    )
end

function NeuralNetworkCache(
        network_parameters::NeuralNetworkParameters,
        kulfan_parameters::KulfanParameters,
        alpha,
        Reynolds
        ;
        n_crit = 9,
        xtr_upper = 1,
        xtr_lower = 1
    )
    return NeuralNetworkCache(
        network_parameters,
        build_features(kulfan_parameters, alpha, Reynolds; n_crit, xtr_upper, xtr_lower)
    )
end
# }}}
# Features processing {{{
# ----------------------------------------------------------------------
"""
    build_features(kulfan_parameters, alpha, Reynolds;
        n_crit=9, xtr_upper=1, xtr_lower=1)

Construct a `25 x N` neural-network input feature array expected by the neural
network from Kulfan shape parameters and flow conditions.

The 25 features are:

- 18 Kulfan parameters (8 upper weights, 8 lower weights, leading edge weight,
  trailing edge thickness scaled by 50)
- `sin(2α)` (α in degrees)
- `cos(α)` (α in degrees)
- `1 - cos²(α)` (α in degrees)
- `(log(Reynolds) - 12.5) / 3.5`
- `(n_crit - 9) / 4.5`
- `xtr_upper`
- `xtr_lower`

# Arguments

- `kulfan_parameters::KulfanParameters`: Airfoil geometry defined by Kulfan
  (CST) parameters.
- `alpha`: Angle of attack in degrees. Can be a scalar or a vector.
- `Reynolds`: Reynolds number. Can be a scalar or a vector.

# Keyword Arguments

- `n_crit::Real=9`: Critical amplification factor used in transition modeling.
- `xtr_upper::Real=1`: Forced transition location on the upper surface (0–1).
- `xtr_lower::Real=1`: Forced transition location on the lower surface (0–1).

# Returns

- `AbstractVector{<:Real}`: Feature vector when `alpha` and `Reynolds` are
  scalars.
- `AbstractMatrix{<:Real}`: Feature matrix of size `(n_features, N)` when
  either `alpha` or `Reynolds` are vectors. Each column corresponds to one
  sample.

# Throws

- `DimensionMismatch`: If `alpha` and `Reynolds` are vectors of different
  lengths.
"""
function build_features(
        kulfan_parameters::KulfanParameters,
        alpha,
        Reynolds;
        n_crit = 9,
        xtr_upper = 1,
        xtr_lower = 1
    )
    _validate_alpha_Reynolds(alpha, Reynolds)
    return _build_features(kulfan_parameters, alpha, Reynolds, n_crit, xtr_upper, xtr_lower)
end

function _build_features(
        kulfan_parameters::KulfanParameters,
        alpha::Real,
        Reynolds::Real,
        n_crit,
        xtr_upper,
        xtr_lower,
    )
    T = promote_type(
        eltype(kulfan_parameters.upper_weights),
        eltype(alpha),
        eltype(Reynolds),
        eltype(n_crit),
        eltype(xtr_upper),
        eltype(xtr_lower),
    )

    x = Vector{T}(undef, 25)

    upper = kulfan_parameters.upper_weights
    lower = kulfan_parameters.lower_weights

    a = T(alpha)
    re = T(Reynolds)
    c = cosd(a)

    @inbounds begin
        for j in 1:8
            x[j] = T(upper[j])
            x[8 + j] = T(lower[j])
        end

        x[17] = T(kulfan_parameters.leading_edge_weight)
        x[18] = T(kulfan_parameters.trailing_edge_thickness) * T(50)
        x[19] = sind(T(2) * a)
        x[20] = c
        x[21] = one(T) - c^2
        x[22] = (log(re) - T(12.5)) / T(3.5)
        x[23] = (T(n_crit) - T(9)) / T(4.5)
        x[24] = T(xtr_upper)
        x[25] = T(xtr_lower)
    end

    return x
end

function _build_features(
        kulfan_parameters::KulfanParameters,
        alpha::AbstractVector{<:Real},
        Reynolds::AbstractVector{<:Real},
        n_crit,
        xtr_upper,
        xtr_lower
    )
    L = length(alpha)

    T = promote_type(
        eltype(kulfan_parameters.upper_weights),
        eltype(alpha),
        eltype(Reynolds),
        eltype(n_crit),
        eltype(xtr_upper),
        eltype(xtr_lower),
    )

    x = Matrix{T}(undef, 25, L)

    upper = kulfan_parameters.upper_weights
    lower = kulfan_parameters.lower_weights

    le = T(kulfan_parameters.leading_edge_weight)
    te = T(kulfan_parameters.trailing_edge_thickness) * T(50)
    ncrit_scaled = (T(n_crit) - T(9)) / T(4.5)
    xtr_u = T(xtr_upper)
    xtr_l = T(xtr_lower)
    c2 = T(2)
    c12_5 = T(12.5)
    c3_5 = T(3.5)

    @inbounds for i in axes(x, 2)
        for j in 1:8
            x[j, i] = T(upper[j])
            x[8 + j, i] = T(lower[j])
        end

        a = T(alpha[i])
        re = T(Reynolds[i])
        c = cosd(a)

        x[17, i] = le
        x[18, i] = te
        x[19, i] = sind(c2 * a)
        x[20, i] = c
        x[21, i] = one(T) - c^2
        x[22, i] = (log(re) - c12_5) / c3_5
        x[23, i] = ncrit_scaled
        x[24, i] = xtr_u
        x[25, i] = xtr_l
    end

    return x
end

_build_features(
    kulfan_parameters::KulfanParameters,
    alpha::AbstractVector{<:Real},
    Reynolds::Real,
    n_crit,
    xtr_upper,
    xtr_lower,
) = _build_features(
    kulfan_parameters, alpha, Fill(Reynolds, length(alpha)), n_crit, xtr_upper, xtr_lower
)

_build_features(
    kulfan_parameters::KulfanParameters,
    alpha::Real,
    Reynolds::AbstractVector{<:Real},
    n_crit,
    xtr_upper,
    xtr_lower,
) = _build_features(
    kulfan_parameters, Fill(alpha, length(Reynolds)), Reynolds, n_crit, xtr_upper, xtr_lower
)

_validate_alpha_Reynolds(::Real, ::Real) = nothing
_validate_alpha_Reynolds(::AbstractVector{<:Real}, ::Real) = nothing
_validate_alpha_Reynolds(::Real, ::AbstractVector{<:Real}) = nothing
function _validate_alpha_Reynolds(
        alpha::AbstractVector{<:Real},
        Reynolds::AbstractArray{<:Real}
    )
    length(alpha) == length(Reynolds) || throw(
        DimensionMismatch("`alpha` and `Reynolds` must have the same length.")
    )
    return nothing
end

"""
    update_features!(cache, x)

Update the input feature arrays stored in the cache.

This function replaces the current input features with `x` and updates the
symmetry-flipped features accordingly.

# Arguments

- `cache::NeuralNetworkCache`: Cache containing input and workspace arrays.
- `x::AbstractArray{<:Real}`: New input feature array of the same size as
  `cache.x`.

# Throws

- `DimensionMismatch`: If `x` does not match the size of the cached input.

# Notes

- This function does not recompute network outputs. After updating the inputs,
  [`evaluate!`](@ref) must be called to obtain updated predictions.
"""
function update_features!(cache::NeuralNetworkCache, x)
    size(x) == size(cache.x) || throw(
        DimensionMismatch(
            "`x` must be of size $(size(cache.x)). An array of size $(size(x)) was given."
        )
    )
    copyto!(cache.x, x)
    copyto!(cache.x_flipped, x)
    flip_inputs!(cache.x_flipped)

    return nothing
end

update_features!(
    cache::NeuralNetworkCache, kulfan_parameters::KulfanParameters,
    alpha, Reynolds;
    n_crit = 9, xtr_upper = 1, xtr_lower = 1
) = update_features!(
    cache; kulfan_parameters, alpha, Reynolds, n_crit, xtr_upper, xtr_lower
)

function update_features!(
        cache::NeuralNetworkCache;
        kulfan_parameters::KulfanParameters,
        alpha,
        Reynolds,
        n_crit = 9,
        xtr_upper = 1,
        xtr_lower = 1
    )
    size(cache.x, 1) == 25 || throw(
        DimensionMismatch(
            "`x` must be of size (25, *). An array of size $(size(cache.x)) was given."
        )
    )
    L = size(cache.x, 2)

    upper = kulfan_parameters.upper_weights
    lower = kulfan_parameters.lower_weights

    length(upper) == 8 || throw(
        DimensionMismatch("`kulfan_parameters.upper_weights` must have length 8.")
    )
    length(lower) == 8 || throw(
        DimensionMismatch("`kulfan_parameters.lower_weights` must have length 8.")
    )

    le = kulfan_parameters.leading_edge_weight
    te = kulfan_parameters.trailing_edge_thickness * 50
    n_crit_scaled = (n_crit - 9) / 4.5

    @inbounds for i in 1:L
        for j in 1:8
            cache.x[j, i] = upper[j]
            cache.x[8 + j, i] = lower[j]
        end

        cache.x[17, i] = le
        cache.x[18, i] = te
    end

    _update_flow_features!(
        cache.x, alpha, Reynolds, L, n_crit_scaled, xtr_upper, xtr_lower
    )

    copyto!(cache.x_flipped, cache.x)
    flip_inputs!(cache.x_flipped)

    return nothing
end

function _update_flow_features!(
        x::AbstractVecOrMat{<:Real},
        alpha::Real,
        Reynolds::Real,
        L::Integer,
        n_crit_scaled,
        xtr_upper,
        xtr_lower
    )
    L == 1 || throw(
        DimensionMismatch(
            "`x` has $L columns, but scalar `alpha` and scalar `Reynolds` define a single sample."
        )
    )

    @inbounds _write_flow_column!(
        x, 1, alpha, Reynolds, n_crit_scaled, xtr_upper, xtr_lower
    )

    return nothing
end

function _update_flow_features!(
        x::AbstractVecOrMat{<:Real},
        alpha::AbstractVector{<:Real},
        Reynolds::Real,
        L::Integer,
        n_crit_scaled,
        xtr_upper,
        xtr_lower
    )
    length(alpha) == L || throw(
        DimensionMismatch(
            "`alpha` must have length $L to match `x`, got length $(length(alpha))."
        )
    )

    @inbounds for i in 1:L
        _write_flow_column!(x, i, alpha[i], Reynolds, n_crit_scaled, xtr_upper, xtr_lower)
    end

    return nothing
end

function _update_flow_features!(
        x::AbstractVecOrMat{<:Real},
        alpha::Real,
        Reynolds::AbstractVector{<:Real},
        L::Integer,
        n_crit_scaled,
        xtr_upper,
        xtr_lower
    )
    length(Reynolds) == L || throw(
        DimensionMismatch(
            "`Reynolds` must have length $L to match `x`, got length $(length(Reynolds))."
        )
    )

    @inbounds for i in 1:L
        _write_flow_column!(x, i, alpha, Reynolds[i], n_crit_scaled, xtr_upper, xtr_lower)
    end

    return nothing
end

function _update_flow_features!(
        x::AbstractVecOrMat{<:Real},
        alpha::AbstractVector{<:Real},
        Reynolds::AbstractVector{<:Real},
        L::Integer,
        n_crit_scaled,
        xtr_upper,
        xtr_lower
    )
    length(alpha) == length(Reynolds) || throw(
        DimensionMismatch("`alpha` and `Reynolds` must have the same length.")
    )
    length(alpha) == L || throw(
        DimensionMismatch(
            "`alpha` and `Reynolds` must have length $L to match `x`, got length $(length(alpha))."
        )
    )

    @inbounds for i in 1:L
        _write_flow_column!(
            x, i, alpha[i], Reynolds[i], n_crit_scaled, xtr_upper, xtr_lower
        )
    end

    return nothing
end

@inline function _write_flow_column!(
        x,
        i::Integer,
        alpha,
        Reynolds,
        n_crit_scaled,
        xtr_upper,
        xtr_lower
    )
    c = cosd(alpha)
    x[19, i] = sind(2 * alpha)
    x[20, i] = c
    x[21, i] = 1 - c^2
    x[22, i] = (log(Reynolds) - 12.5) / 3.5
    x[23, i] = n_crit_scaled
    x[24, i] = xtr_upper
    x[25, i] = xtr_lower

    return nothing
end

"""
    flip_inputs!(x)

Flip the input array in-place, creating a geometrically mirrored version of the
input features.

# Arguments

- `x::AbstractArray`: Input array of size (25, *) where each column represents
  a sample. Flipping is applied on specific rows.
"""
function flip_inputs!(x)
    size(x, 1) == 25 || throw(
        DimensionMismatch(
            "`x` must be of size (25, *). An array of size $(size(x)) was given."
        )
    )

    @inbounds for i in axes(x, 2)
        for j in 1:8
            x[j, i], x[(8 + j), i] = -x[(8 + j), i], -x[j, i]
        end

        x[17, i] *= -1
        x[19, i] *= -1
        x[(end - 1), i], x[end, i] = x[end, i], x[(end - 1), i]
    end

    return nothing
end
# }}}
# Neural network {{{
# ----------------------------------------------------------------------
"""
    forward(network_parameters::NetworkParameters, x::AbstractMatrix{<:Real})

Evaluate the neural network using the pretrained network parameters on the
given input `x`.

# Arguments

- `network_parameters::NeuralNetworkParameters`: pretrained network weights and
  biases.
- `x::AbstractArray{<:Real}`: Input data of size (25, :).

# Returns

- `AbstractMatrix{<:Real}`
"""
function forward(network_parameters::NeuralNetworkParameters, x::AbstractArray{<:Real})
    weights = network_parameters.weights
    biases = network_parameters.biases

    @inbounds for (i, (W, b)) in enumerate(zip(weights, biases))
        x = muladd(W, x, b)

        if i != length(weights)
            x = swish.(x)
        end
    end

    return x
end

"""
    forward!(y, network_parameters, x)

Evaluate the neural network in-place using preallocated buffers.

# Arguments

- `y::AbstractArray{<:Real}`: Output array that will store the network
  predictions.
- `network_parameters::NeuralNetworkParameters`: Pretrained network weights
  and biases.
- `x::AbstractVector{<:}`: Temporary cache, where `x[1]` contains the input
  features and subsequent entries store intermediate layer outputs.

# Notes

- Use [`allocate_forward_cache`](@ref) to create `y` and `x` caches.
"""
function forward!(y, network_parameters, x)
    weights = network_parameters.weights
    biases = network_parameters.biases

    @inbounds for i in axes(weights, 1)
        W = weights[i]
        b = biases[i]

        LinearAlgebra.mul!(x[i + 1], W, x[i])
        x[i + 1] .= x[i + 1] .+ b

        if i < length(weights)
            x[i + 1] .= swish.(x[i + 1])
        end
    end

    y .= x[end]

    return nothing
end

"""
    swish(x, β=1)

Swish activation function.
"""
@inline swish(x, β = one(x)) = x * inv(one(x) + exp(-β * x))

"""
    sigmoid(x)

Sigmoid activation function with input clipping for numerical stability.
"""
@inline function sigmoid(x)
    ln_eps = _ln_eps(x)
    return inv(one(x) + exp(-clamp(x, ln_eps, -ln_eps)))
end

@inline _ln_eps(::T) where {T <: Real} = log(T(10) / floatmax(T))

"""
    allocate_forward_cache(network_parameters, x)

Allocate output and activation buffers for in-place neural network evaluation.

# Arguments

- `network_parameters::NeuralNetworkParameters`: Pretrained network weights
  and biases.
- `x::AbstractVecOrMat{<:Real}`: Input features used to determine the shape
  of the allocated buffers.

# Returns

- `y`: Output array matching the network output dimensions.
- `x`: Vector of arrays storing temporary cache, where `cache[1]` corresponds
  to the input and subsequent entries store intermediate results.
"""
function allocate_forward_cache(
        network_parameters::NeuralNetworkParameters,
        x::AbstractVecOrMat{<:Real}
    )
    y = zeros(size(network_parameters.weights[end], 1), size(x, 2))
    z(w) = x isa AbstractVector ? zeros(size(w, 1)) : zeros(size(w, 1), size(x, 2))

    return (y, [[x]; [z(w) for w in network_parameters.weights]])
end
# }}}
# Evaluation {{{
# ----------------------------------------------------------------------
"""
    evaluate(network_parameters, x) -> NeuralNetworkOutput

Evaluate the neural network for the input features `x` using the pretrained
parameters `network_parameters`. It then applies post-processing
transformations to produce physically meaningful aerodynamic coefficients.

# Arguments

- `network_parameters::NeuralNetworkParameters`: Pretrained parameters of the
  neural network.
- `x::AbstractMatrix`: Matrix of preprocessed features characterizing the
  airfoil geometry and the flow conditions. Each column corresponds to one
  input sample.

# Returns

- [`NeuralNetworkOutput`](@ref): Predicted aerodynamic coefficients.
"""
function evaluate(network_parameters, x)
    y = forward(network_parameters, x)
    confidence_correction!(y, x, network_parameters)

    x_flipped = copy(x)
    flip_inputs!(x_flipped)
    y_flipped = forward(network_parameters, x_flipped)
    confidence_correction!(y_flipped, x_flipped, network_parameters)

    tmp = y_flipped[6, :]
    @views begin
        flip_outputs!(y_flipped, tmp)
        fuse_predictions!(y, y_flipped)
        decode_outputs!(y)
    end

    return pack_output(y)
end

function evaluate(
        network_parameters::NeuralNetworkParameters,
        kulfan_parameters::KulfanParameters,
        alpha,
        Reynolds
        ;
        n_crit = 9,
        xtr_upper = 1,
        xtr_lower = 1
    )
    return evaluate(
        network_parameters,
        build_features(kulfan_parameters, alpha, Reynolds; n_crit, xtr_upper, xtr_lower)
    )
end

"""
    evaluate!(cache)

Evaluate the neural network in-place using a preallocated cache.

# Arguments

- `cache::NeuralNetworkCache`: Cache containing input features, intermediate
  buffers, and output storage.

# Notes

- Build the cache using the [`NeuralNetworkCache`](@ref) constructor.
- Use [`update_features!`](@ref) to modify inputs before evaluation.

# See also

- [`evaluate`](@ref): out-of-place version that allocates outputs and
  intermediate computations.
"""
function evaluate!(cache::NeuralNetworkCache)
    @views begin
        # NOTE: a two-pass approach is faster for scalar inputs, whereas a
        # concatenated pass is faster for batched inputs.
        if cache.x isa AbstractVector
            forward!(cache.y, cache.network_parameters, cache.tmp_x)
            forward!(cache.y_flipped, cache.network_parameters, cache.tmp_x_flipped)
        else
            forward!(cache.y_both, cache.network_parameters, cache.tmp_x_both)
        end

        confidence_correction!(cache.y, cache.x, cache)
        confidence_correction!(cache.y_flipped, cache.x_flipped, cache)

        flip_outputs!(cache.y_flipped, cache.tmp_y_smd[:, 1])
        fuse_predictions!(cache.y, cache.y_flipped)
        decode_outputs!(cache.y)
        pack_output!(cache.outputs, cache.y)
    end

    return nothing
end

"""
    squared_mahalanobis_distance(params::NetworkParameters, x)

Compute the squared Mahalanobis distance between the input array `x` and the
mean of the scaled input distribution.

# Arguments

- `params::NetworkParameters`: pretrained neural network parameters containing
  the mean and inverse covariance of the scaled input distribution.
- `x::AbstractArray{<:Real}`: Input samples.

# Returns

- `AbstractArray{<:Real}`
"""
function squared_mahalanobis_distance(
        params::NeuralNetworkParameters,
        x::AbstractMatrix{<:Real}
    )
    x_minus_mean = x .- params.mean_inputs_scaled

    return sum(
        x_minus_mean .* (params.inv_cov_inputs_scaled * x_minus_mean);
        dims = 1
    )'
end

function squared_mahalanobis_distance(
        params::NeuralNetworkParameters,
        x::AbstractVector{<:Real},
    )
    x_minus_mean = x .- params.mean_inputs_scaled

    return LinearAlgebra.dot(x_minus_mean, params.inv_cov_inputs_scaled * x_minus_mean)
end

"""
    squared_mahalanobis_distance!(y, params::NetworkParameters, x, tmp1, tmp2)

In-place, non-allocating version of [`squared_mahalanobis_distance`](@ref).

Compute the squared Mahalanobis distance between the input array `x` and the
mean of the scaled input distribution.

# Arguments

- `y::AbstractVector{<:Real}`: Output vector of size `size(x, 2)`.
- `params::NetworkParameters`: pretrained neural network parameters containing
  the mean and inverse covariance of the scaled input distribution.
- `x::AbstractArray{<:Real}`: Input samples.
- `tmp1::AbstractArray{<:Real}`: Temporary array the same size as `x`.
- `tmp2::AbstractArray{<:Real}`: Temporary array the same size as `x`.
"""
function squared_mahalanobis_distance!(y, params::NeuralNetworkParameters, x, tmp1, tmp2)
    tmp1 .= x .- params.mean_inputs_scaled

    LinearAlgebra.mul!(tmp2, params.inv_cov_inputs_scaled, tmp1)
    tmp1 .= tmp1 .* tmp2

    @views @inbounds for i in axes(tmp1, 2)
        sum!(y[i, :], tmp1[:, i])
    end

    return nothing
end

"""
    confidence_correction!(y, x, network_parameters)
    confidence_correction!(y, x, cache)

Apply Mahalanobis-distance-based confidence correction to the network outputs.

This function adjusts the first output channel (confidence) based on the
distance between the input features and the training data distribution.

# Arguments

- `y::AbstractArray{<:Real}`: Network output array. The first row is modified
  in-place.
- `x::AbstractArray{<:Real}`: Input feature array.
- `network_parameters::NeuralNetworkParameters`: Pretrained network parameters
- `cache::NeuralNetworkCache`: Cache containing preallocated arrays

# Notes

- The cache-based method avoids intermediate allocations and is preferred in
  performance-critical loops.
"""
function confidence_correction!(
        y::AbstractMatrix{<:Real},
        x::AbstractMatrix{<:Real},
        network_parameters::NeuralNetworkParameters
    )
    @views y[1, :] .-= (
        squared_mahalanobis_distance(network_parameters, x) ./ (2 * size(x, 1))
    )
    return nothing
end

function confidence_correction!(
        y::AbstractVector{<:Real},
        x::AbstractVector{<:Real},
        network_parameters::NeuralNetworkParameters,
    )
    y[1] -= squared_mahalanobis_distance(network_parameters, x) / (2 * length(x))
    return nothing
end

function confidence_correction!(y, x, cache::NeuralNetworkCache)
    squared_mahalanobis_distance!(
        cache.tmp_y_smd,
        cache.network_parameters,
        x,
        cache.tmp_x_smd1,
        cache.tmp_x_smd2
    )
    @views y[1, :] .-= cache.tmp_y_smd[:, 1] ./ (2 * size(x, 1))
    return nothing
end

"""
    flip_outputs!(y, tmp)

Transform neural network outputs so that they are consistent with the
original (non-flipped) reference frame.

# Arguments

- `y::AbstractArray{<:Real}`: Output array to be transformed in-place.
- `tmp::AbstractVector{<:Real}`: Temporary array used to swap outputs.
"""
function flip_outputs!(y, tmp)
    @views begin
        tmp .= y[6, :]
        y[2, :] .= (-).(y[2, :])
        y[4, :] .= (-).(y[4, :])
        y[6, :] .= y[5, :]
        y[5, :] .= tmp
    end
    return nothing
end

"""
    fuse_predictions!(y, y_flipped)

Average predictions from original and symmetry-flipped inputs.

# Arguments

- `y::AbstractArray{<:Real}`: Output array that will store the fused result.
- `y_flipped::AbstractArray{<:Real}`: Output array from flipped inputs.

# Notes

- The result overwrites `y`.
- Both arrays must have identical sizes.
"""
function fuse_predictions!(y, y_flipped)
    y .= y .+ y_flipped
    y .= y ./ 2
    return nothing
end

"""
    decode_outputs!(y)

Transform raw neural network outputs into physically meaningful quantities.

This function applies scaling, nonlinear transformations, and clamping to
convert network outputs into aerodynamic coefficients and transition
locations.

# Arguments

- `y::AbstractArray{<:Real}`: Network output array modified in-place.

!!! note
    The ordering and scaling of outputs are part of the trained model and must
    not be modified without retraining the network.
"""
function decode_outputs!(y)
    @views begin
        y[1, :] .= sigmoid.(y[1, :])
        y[2, :] ./= 2
        y[3, :] .= exp.((y[3, :] .- 2) .* 2)
        y[4, :] ./= 20
        y[5, :] .= clamp.(y[5, :], 0, 1)
        y[6, :] .= clamp.(y[6, :], 0, 1)
    end
    return nothing
end

"""
    pack_output(y) -> NeuralNetworkOutput

Convert a decoded output array into a `NeuralNetworkOutput` struct.

# Arguments

- `y::AbstractArray{<:Real}`: Decoded network output array.

# Returns

- [`NeuralNetworkOutput`](@ref)
"""
function pack_output(y::AbstractMatrix{<:Real})
    return @views NeuralNetworkOutput(
        analysis_confidence = y[1, :],
        CL = y[2, :],
        CD = y[3, :],
        CM = y[4, :],
        Top_Xtr = y[5, :],
        Bot_Xtr = y[6, :]
    )
end

function pack_output(y::AbstractVector{<:Real})
    return NeuralNetworkOutput(
        analysis_confidence = y[1],
        CL = y[2],
        CD = y[3],
        CM = y[4],
        Top_Xtr = y[5],
        Bot_Xtr = y[6],
    )
end

"""
    pack_output!(output, y)

Store decoded outputs into a preallocated `NeuralNetworkOutput` struct.

# Arguments

- `output::NeuralNetworkOutput`: Output container to be updated.
- `y::AbstractArray{<:Real}`: Decoded network output array.
"""
function pack_output!(output::NeuralNetworkOutput, y)
    @views begin
        output.analysis_confidence .= y[1, :]
        output.CL .= y[2, :]
        output.CD .= y[3, :]
        output.CM .= y[4, :]
        output.Top_Xtr .= y[5, :]
        output.Bot_Xtr .= y[6, :]
    end
    return nothing
end
# }}}

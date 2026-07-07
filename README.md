<h1 align="center">
    NNFoil.jl
</h1>

<div align="center">

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gabrielbdsantos.github.io/NNFoil.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gabrielbdsantos.github.io/NNFoil.jl/dev/)
[![Build Status](https://github.com/gabrielbdsantos/NNFoil.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gabrielbdsantos/NNFoil.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

</div>

NNFoil.jl is a Julia implementation of
[NeuralFoil](https://github.com/peterdsharpe/NeuralFoil/), a physics-informed
machine-learning model for airfoil aerodynamic analysis. The package began as
an effort to address a correctness issue in
[NeuralFoil.jl](https://github.com/byuflowlab/NeuralFoil.jl/) (see this
[issue](https://github.com/byuflowlab/NeuralFoil.jl/issues/5) for details), and
has since grown into a more accurate and performant implementation.

NNFoil.jl is tested against the original Python package using more than 1,600
airfoil samples from the [UIUC Airfoil Coordinates
Database](https://m-selig.ae.illinois.edu/ads/coord_database.html) to ensure
consistent predictions.

## Installation

To install NNFoil.jl:

1. Download [Julia](https://julialang.org/downloads/) version 1.10 or later.
1. Launch Julia and run:

```julia-repl
julia> import Pkg
julia> Pkg.add("NNFoil")
```

To install the latest development version:

```julia-repl
julia> import Pkg
julia> Pkg.add(url = "https://github.com/gabrielbdsantos/NNFoil.jl")
```

## Quick Start

```julia
using NNFoil
using DelimitedFiles

coordinates = readdlm(
    abspath(
        joinpath(
            NNFoil.DATA_PATH, splitpath("../test/airfoils/raw/naca0018.dat")...
        )
    )
)
kulfan_parameters = KulfanParameters(normalize_coordinates!(coordinates))
network_parameters = NeuralNetworkParameters(; model_size = :xlarge)
alpha = -180:180
Reynolds = 5.0e6

analysis = evaluate(network_parameters, kulfan_parameters, alpha, Reynolds)
```

For repeated evaluations, use a cache to reuse preallocated buffers:

```julia
using NNFoil
using DelimitedFiles

coordinates = readdlm(
    abspath(
        joinpath(
            NNFoil.DATA_PATH, splitpath("../test/airfoils/raw/naca0018.dat")...
        )
    )
)
kulfan_parameters = KulfanParameters(normalize_coordinates!(coordinates))
network_parameters = NeuralNetworkParameters(; model_size = :xlarge)
alpha = -180:180
Reynolds = 5.0e6

cache = NeuralNetworkCache(network_parameters, kulfan_parameters, alpha, Reynolds)
evaluate!(cache)
analysis = cache.outputs

Reynolds_updated = 7.5e6
update_features!(cache, kulfan_parameters, alpha, Reynolds_updated)
evaluate!(cache)
analysis_updated = cache.outputs
```

## Automatic Differentiation

NNFoil.jl supports automatic differentiation through the neural-network
evaluation path. The out-of-place API (`evaluate`) and the cached in-place API
(`evaluate!`) are tested with ForwardDiff, Enzyme, and Mooncake (experimental)
against finite differences.

Use the same objective-function style you would use with other Julia AD tools:

```julia
using NNFoil
using ForwardDiff

network_parameters = NeuralNetworkParameters(; model_size = :xsmall, T = Float64)
x = build_features(kulfan_parameters, alpha, Reynolds)

objective(v) = begin
    features = reshape(v, size(x))
    outputs = evaluate(network_parameters, features)
    return sum(outputs.CL ./ outputs.CD)
end

gradient = ForwardDiff.gradient(objective, vec(copy(x)))
```

For repeated differentiable evaluations, make sure `NeuralNetworkCache` is
allocated with an element type compatible with the AD input. For ForwardDiff,
the simplest pattern is to construct the cache inside the differentiated
objective. See `examples/example03_forwarddiff.jl` for a complete ForwardDiff
example.

## Citing

If you use NNFoil.jl in your research, please cite both the
[original Python package](https://github.com/peterdsharpe/NeuralFoil)

```bibtex
@misc{neuralfoil,
  author = {Peter Sharpe},
  title = {{NeuralFoil}: An airfoil aerodynamics analysis tool using physics-informed machine learning},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/peterdsharpe/NeuralFoil}},
}
```

and
[the original author's PhD thesis](https://dspace.mit.edu/handle/1721.1/157809),
which includes an extended chapter that serves as the primary long-form
documentation for the tool:

```bibtex
@phdthesis{aerosandbox_phd_thesis,
   title = {Accelerating Practical Engineering Design Optimization with Computational Graph Transformations},
   author = {Sharpe, Peter D.},
   school = {Massachusetts Institute of Technology},
   year = {2024},
}
```

## Acknowledgments

Special thanks to Judd Mehr from the [BYU FLOW Lab](https://flow.byu.edu) for
putting together an earlier translation of the original Python package to
Julia.

## License

NNFoil.jl is released under the terms of the MIT license. See the
[LICENSE](./LICENSE) file for details.

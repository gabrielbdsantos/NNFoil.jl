# NNFoil.jl

This is a partial Julia translation of
[NeuralFoil](https://github.com/peterdsharpe/NeuralFoil/) inspired by
[NeuralFoil.jl](https://github.com/byuflowlab/NeuralFoil.jl/). NNFoil.jl has
been thoroughly tested against the original Python package using a database of
more than 1600 airfoil samples to ensure a consistent implementation. See this
[issue](https://github.com/byuflowlab/NeuralFoil.jl/issues/5) for more
information on how NNFoil.jl differs from NeuralFoil.jl.

## Installation

NNFoil.jl is not yet a registered Julia package. So to install it,

1. Download [Julia](https://julialang.org/downloads/) version 1.10 or later.
1. Launch Julia and type

```julia-repl
julia> import Pkg
julia> Pkg.add("https://github.com/gabrielbdsantos/NNFoil.jl")
```

## Quick start

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

## In-place evaluation

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
update_features!(
    cache;
    kulfan_parameters,
    alpha,
    Reynolds = Reynolds_updated,
)
evaluate!(cache)
analysis_updated = cache.outputs
```

See `examples/example02.jl` for a complete in-place example.

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
which has an extended chapter that serves as the primary long-form
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

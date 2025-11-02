# NNFoil.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gabrielbdsantos.github.io/NNFoil.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gabrielbdsantos.github.io/NNFoil.jl/dev/)
[![Build Status](https://github.com/gabrielbdsantos/NNFoil.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gabrielbdsantos/NNFoil.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)


This is a partial Julia translation of
[NeuralFoil](https://github.com/peterdsharpe/NeuralFoil/) inspired by
[NeuralFoil.jl](https://github.com/byuflowlab/NeuralFoil.jl/). NNFoil.jl has
been thoroughly tested against the original Python package using a database of
more than 1600 airfoil samples to ensure a consistent implementation. See this
[issue](https://github.com/byuflowlab/NeuralFoil.jl/issues/5) for more
information on how NNFoil.jl differs from NeuralFoil.jl.

## Installation

```julia-repl
julia> import Pkg
julia> Pkg.add("https://github.com/gabrielbdsantos/NNFoil.jl")
```

## Quick Start

```julia
import NNFoil
import DelimitedFiles

coordinates = DelimitedFiles.readdlm(joinpath(
    NNFoil.DATA_PATH, split("../test/airfoils/raw/naca0018.dat", "/")...
))
kulfan_parameters = NNFoil.KulfanParameters(NNFoil.normalize_coordinates!(coordinates))
network_parameters = NNFoil.NeuralNetworkParameters(; model_size = :xlarge)
alpha = -180:180
Reynolds = 5e6

analysis = NNFoil.evaluate(network_parameters, kulfan_parameters, alpha, Reynolds)
```

## Citing

If you use NNFoil.jl in your research, please cite both the [original Python
package](https://github.com/peterdsharpe/NeuralFoil)

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

and [the original author's PhD
thesis](https://dspace.mit.edu/handle/1721.1/157809), which has an extended
chapter that serves as the primary long-form documentation for the tool:

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

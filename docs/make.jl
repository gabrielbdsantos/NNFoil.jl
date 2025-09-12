using NNFoil
using Documenter

DocMeta.setdocmeta!(NNFoil, :DocTestSetup, :(using NNFoil); recursive=true)

makedocs(;
    modules=[NNFoil],
    authors="Gabriel B. dos Santos <gabriel.bertacco@unesp.br>",
    sitename="NNFoil.jl",
    format=Documenter.HTML(;
        canonical="https://gabrielbdsantos.github.io/NNFoil.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gabrielbdsantos/NNFoil.jl",
    devbranch="main",
)

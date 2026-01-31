using NNFoil
using Documenter

DocMeta.setdocmeta!(NNFoil, :DocTestSetup, :(using NNFoil); recursive = true)

makedocs(;
    modules = [NNFoil],
    authors = "Gabriel B. dos Santos <gabriel.bertacco@unesp.br>",
    sitename = "NNFoil.jl",
    format = Documenter.HTML(;
        canonical = "https://gabrielbdsantos.github.io/NNFoil.jl",
        edit_link = "main",
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API" => "public-interface.md",
    ],
)

deploydocs(;
    repo = "github.com/gabrielbdsantos/NNFoil.jl",
    devbranch = "main",
)

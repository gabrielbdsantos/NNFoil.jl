try
    # Try to reuse the local installed conda environment
    readdir(".CondaPkg")

    ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
    ENV["JULIA_PYTHONCALL_EXE"] = joinpath(
        pwd(),
        split("/.CondaPkg/.pixi/envs/default/bin/python", "/")...
    )
catch IOError
    # Otherwise create a new Python environment and install the necessary packages
    using CondaPkg
    CondaPkg.add_pip("aerosandbox")
end

using PythonCall

const py_array = pyimport("numpy").array
const py_genfromtxt = pyimport("numpy").genfromtxt
const py_get_kulfan_parameters =
    pyimport("aerosandbox.geometry.airfoil.airfoil_families").get_kulfan_parameters
const py_get_aero_from_kulfan_parameters =
    pyimport("neuralfoil").get_aero_from_kulfan_parameters
const Airfoil = pyimport("aerosandbox.geometry.airfoil").Airfoil

function py_get_kulfan_from_coordinates(coordinates; normalize_coordinates = false)
    return py_get_kulfan_parameters(coordinates, normalize_coordinates = normalize_coordinates)
end

function py_get_kulfan_from_file(filepath; normalize_coordinates = false)
    coordinates = py_array(coordinates_from_file(filepath))
    return py_get_kulfan_from_coordinates(coordinates; normalize_coordinates)
end

function convert_kulfan_py2jl(params)
    upper_weights = pyconvert(Vector{Float64}, params["upper_weights"])
    lower_weights = pyconvert(Vector{Float64}, params["lower_weights"])
    leading_edge_weight = pyconvert(Float64, params["leading_edge_weight"])
    trailing_edge_thickness = pyconvert(Float64, params["TE_thickness"])

    return KulfanParameters(
        upper_weights = upper_weights,
        lower_weights = lower_weights,
        leading_edge_weight = leading_edge_weight,
        trailing_edge_thickness = trailing_edge_thickness,
    )
end

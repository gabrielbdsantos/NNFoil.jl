try
    # Try to reuse the local installed conda environment
    readdir(".CondaPkg")

    @info "Using local `.CondaPkg`"

    ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
    ENV["JULIA_PYTHONCALL_EXE"] = joinpath(pwd(), split("/.CondaPkg/.pixi/envs/default/bin/python", "/")...)
catch IOError
    # Otherwise create a new Python environment and install the necessary packages
    using CondaPkg
    CondaPkg.add_pip("aerosandbox")
end


using PythonCall


macro wrap_pyfunction(mod, fname, jname)
    quote
        const pymod = pyimport($mod)

        # Define functions with the same names as the input symbols
        $(:(
            function $(esc(jname))(args...; kwargs...)
                pyf = @pyconst pymod.$(fname)
                return pyf(args...; kwargs...)
            end
        ))
    end
end


@wrap_pyfunction "numpy" array py_array
@wrap_pyfunction "numpy" genfromtxt py_genfromtxt
@wrap_pyfunction "aerosandbox.geometry.airfoil.airfoil_families" get_kulfan_parameters py_get_kulfan_parameters


function py_get_kulfan_from_coordinates(coordinates; normalize_coordinates=false)
    params = py_get_kulfan_parameters(coordinates, normalize_coordinates=normalize_coordinates)

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


function py_get_kulfan_from_file(filepath; normalize_coordinates=false)
    coordinates = np_array(coordinates_from_file(filepath))
    return py_get_kulfan_from_coordinates(coordinates; normalize_coordinates)
end

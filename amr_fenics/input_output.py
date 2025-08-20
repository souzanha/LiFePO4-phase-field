# -----------------------------------------------------------------------------
# Copyright (c) 2022 Nana Ofori-Opoku. All rights reserved.

import os
import dolfin as df
import amr_fenics.helpers as helpers
import amr_fenics.mesh_domain_funcs as mdf

try:
    from rich import print
except ImportError:
    pass


def disp(*args, **kwargs):
    """String output/display on headnode.

    Parameters
    ----------
    String : string
        string to be outputed and displayed

    Returns
    -------
    print output of string on headnode

    """
    if helpers.isHead:
        print(*args, **kwargs)


def output_directory(dir):
    """Initializes the directory for the simulations run and file handle.

    Parameters
    ----------
    dir : string
        name of directory to house simulation output files.

    Returns
    -------
    object
        File handle for output files.

    """
    if helpers.isHead:
        if not os.path.exists(dir):
            os.makedirs(dir)

    return initialize_output(dir)


def initialize_output(dir):
    """Initializes the output file system. xdmf.

    Parameters
    ----------
    dir : string
        name of directory to house simulation output files.

    Returns
    -------
    object
        File handle for output files.

    """

    file_solution = df.XDMFFile(helpers.mpiComm, "%s/%s.xdmf" % (dir, "solution"))
    # allows us to view solution while still executing
    file_solution.parameters["flush_output"] = True
    # Is the mesh changing
    file_solution.parameters["rewrite_function_mesh"] = True
    # do all functions share the same mesh
    file_solution.parameters["functions_share_mesh"] = True

    return file_solution


def write_fields(U, t, names, fs):
    """Writes data to initialised xdmf.

    Parameters
    ----------
    U : FEniCS function. List
            Holds all simulation variables

    t : Float
            Current simulation timestamp

    names: Array/List of strings
            An array of names for each simulation variable.
            Same length as FEniCS functions.

    fs : Object
        File object that was defined to house output.

    Returns
    -------
    Nothing

    """

    sol = U.split()
    fields = {names[r]: sol[r] for r in range(len(sol))}

    for name, field in fields.items():
        field.rename(name, name)
        fs.write(field, t)
    fs.close()


def write_aux(fn, t, name, V, fs):
    """Writes auxiliary simulation data to file.
       That is intermediary data or functions.

    Parameters
    ----------
    fn : FEniCS function.
            Auxiliary fenics function.

    t : Float
            Current simulation timestamp

    name: String
            Name string for auxiliary variable.

    V: FEniCS df.FunctionSpace
            Dimensional space/domain which function can be projected onto.

    fs : Object
        File object that was defined to house output.

    Returns
    -------
    Nothing

    """
    s = df.project(fn, V)
    s.rename(name, name)
    fs.write(s, t)
    fs.close()


def output_restart(dir, imesh, mesh, U, t, dt):
    """Checking pointing simulation.
    Outputs set of files so simulation can be restarted.

    Parameters
    ----------
    dir : Object
        Directory object for file storage.

    mesh: FEniCS mesh function
            Defines the simulation mesh and domain.

    U : FEniCS function. List.
            Holds all simulation variables

    Returns
    -------
    Nothing

    """

    restart_solution = df.HDF5File(helpers.mpiComm, dir + "/restart.h5", "w")
    restart_solution.write(imesh, "/imesh")
    restart_solution.write(mesh, "/mesh")
    restart_solution.write(U, "/fields")
    attr = restart_solution.attributes("/fields")
    attr["time"] = t
    attr["timestep"] = dt(0)
    restart_solution.flush()
    restart_solution.close()


def read_restart(run, dt, nprev, PB, orgmesh):
    """Reads the files to restart simulation.
    Constructs previous finite element object, mesh and FEniCS functions.

    Parameters
    ----------
    dir : Object
        Directory object for file storage.

    Returns
    -------
    P1 : FEniCS finite element object.
            Finite element.

    """

    restart_solution = df.HDF5File(helpers.mpiComm, run + "/restart.h5", "r")
    mesh = df.Mesh()
    restart_solution.read(mesh, "/mesh", False)

    nmesh = df.Mesh(mesh)

    P1 = df.FiniteElement("Lagrange", orgmesh.ufl_cell(), 1)
    Pv = df.VectorElement("Lagrange", orgmesh.ufl_cell(), 1)
    element = df.MixedElement([P1, P1, Pv])

    U, U_n, V = mdf.set_functions(nmesh,element,nprev,PB)

    restart_solution.read(U, "/fields")
    attr = restart_solution.attributes("/fields")
    t = attr["time"]
    dt.assign(attr["timestep"])

    restart_solution.close()

    disp("restart time = {}, last recorded dt = {}".format(t, dt(0)))

    return P1, element, V, nmesh, U, U_n, t, dt


def output_mesh(dir, mesh):

    restart_mesh = df.HDF5File(helpers.mpiComm, dir + "/mesh.h5", "w")
    restart_mesh.write(mesh, "/mesh")
    restart_mesh.close()


def read_mesh(dir):

    restart_mesh = df.HDF5File(helpers.mpiComm, dir + "/mesh.h5", "r")
    mesh = df.Mesh()
    restart_mesh.read(mesh, "/mesh", False)
    restart_mesh.close()

    return mesh


def output_avg_stats(filen, *arg):

    if helpers.isHead:  # Output energy file
        f = open(filen, "a")
        f.write(",".join([str(s) for s in list(arg)]))
        f.write("\n")
        f.close()

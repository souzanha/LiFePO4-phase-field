# -----------------------------------------------------------------------------
# Copyright (c) 2022 Nana Ofori-Opoku. All rights reserved.

"""This module imports fenics and python specific modules.
Imports:
        ufl, dolfin, fenics
        numpy, time
        matplotlib
        gc
        tqdm

Functions:

"""
import os
import dolfin as df
import numpy as np
import tqdm
import warnings

import amr_fenics.helpers as helpers
import amr_fenics.input_output as io

try:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import matplotlib.pyplot as plt

    def show_tmp(U):
        fields = U.split()
        for r in range(len(fields)):
            plt.figure()
            plt.plot(fields[r])
            plt.colorbar()

        plt.show()

except ImportError:
    pass


def memory_usage(as_string=True):
    """
    Return memory usage of current process. Note that this function
    only works on Linux systems. If the flag as_string is true, memory
    usage is returned as a string. Otherwise, a tuple of integers is
    returned with current memory usage and maximum memory usage in GB.
    """

    # Try reading /proc
    pid = os.getpid()
    with open("/proc/%d/status" % pid) as fh:
        # Parse values
        status = fh.read()
        vmsize = int(status.split("VmSize:")[1].split("kB")[0]) / 1048576.0
        vmpeak = int(status.split("VmPeak:")[1].split("kB")[0]) / 1048576.0

    # Return values
    if as_string:
        io.disp(
            "Memory usage is {:.2f} GB (peak usage is {:.2g} GB)".format(vmsize, vmpeak)
        )

    return (vmsize, vmpeak)


def field_average(U, val):
    fields = U.split()
    return df.assemble(fields[val] * df.dx)


def integrate_func(Func):
    return df.assemble(Func * df.dx)


def L2_error(U, U_n, mesh, element, *val):
    V = df.FunctionSpace(mesh, element)

    u = []
    u_old = []
    error = np.zeros(len(val))

    [u.append(df.Function(V)) for i in range(len(val))]
    [u_old.append(df.Function(V)) for i in range(len(val))]

    [df.assign(u[i], U.sub(val[i])) for i in range(len(val))]
    [df.assign(u_old[i], U_n.sub(val[i])) for i in range(len(val))]

    error = [
        df.errornorm(u[i], u_old[i], norm_type="l2", degree_rise=3, mesh=mesh)
        for i in range(len(val))
    ]
    io.disp("L2 error for field are {}".format(error))
    return all(err < 1e-8 for err in error)


def equil_gradient(U, mesh, element, tol, *val):
    V = df.FunctionSpace(mesh, element)

    u = []
    error = np.zeros(len(val))

    [u.append(df.Function(V)) for i in range(len(val))]

    [df.assign(u[i], U.sub(val[i])) for i in range(len(val))]

    msize = df.assemble(
        df.interpolate(df.Constant(1.0), df.FunctionSpace(mesh, element)) * df.dx
    )

    error = [
        df.assemble(df.sqrt(df.inner(df.grad(u[i]), df.grad(u[i]))) * df.dx) / msize
        for i in range(len(val))
    ]
    io.disp(f"│  ├───── Gradient for equilibrium fields are: {error}")
    return all(err < tol for err in error)


def H10_error(U, U_n, mesh, element, *val):
    V = df.FunctionSpace(mesh, element)

    u = []
    u_old = []
    error = np.zeros(len(val))

    [u.append(df.Function(V)) for i in range(len(val))]
    [u_old.append(df.Function(V)) for i in range(len(val))]

    [df.assign(u[i], U.sub(val[i])) for i in range(len(val))]
    [df.assign(u_old[i], U_n.sub(val[i])) for i in range(len(val))]

    error = [
        df.errornorm(u[i], u_old[i], norm_type="H10", degree_rise=3, mesh=mesh)
        for i in range(len(val))
    ]
    io.disp("H10 error for fields are {}".format(error))
    return all(err < 1e-6 for err in error)


def noise(U, iter, type):
    if type == 0:
        np.random.seed(iter + helpers.mpiRank)
        size = U.vector().local_size()

        return np.random.normal(0, 1, size)

    if type == 1:
        np.random.seed(iter + helpers.mpiRank)
        size = U.vector().local_size()

        return np.random.uniform(0, 1, size)

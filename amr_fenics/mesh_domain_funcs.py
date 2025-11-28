# -----------------------------------------------------------------------------
# Copyright (c) 2022 Nana Ofori-Opoku. All rights reserved.

import dolfin as df
import numpy as np
import amr_fenics.input_output as io

mpiComm = df.MPI.comm_world
mpiRank = df.MPI.rank(mpiComm)
verbose = True


def within_eps(a, b):
    return bool(a < b + df.DOLFIN_EPS) and bool(a > b - df.DOLFIN_EPS)


def xboundary(L):
    return df.CompiledSubDomain("near(x[0], Lx, 1e-14) && on_boundary", Lx=L)


def yboundary(L):
    return df.CompiledSubDomain("near(x[1], Ly, 1e-14) && on_boundary", Ly=L)


def zboundary(L):
    return df.CompiledSubDomain("near(x[2], Lz, 1e-14) && on_boundary", Lz=L)


class PeriodicBoundary1D(df.SubDomain):
    def __init__(self, Lx, **kwargs):
        df.SubDomain.__init__(self)
        self.Lx = Lx
        super().__init__(**kwargs)

        if verbose:
            print("[{}] PeriodicBoundary1D instantiated.".format(mpiRank))

    def inside(self, x, on_boundary):
        # boundary minus the right side
        return on_boundary and bool(df.near(x[0], 0) and not df.near(x[0], self.Lx))

    def map(self, x, y):
        # Map points from right boundary to left reference
        if df.near(x[0], self.Lx):
            y[0] = x[0] - self.Lx
        else:
            y[0] = x[0]


class PeriodicBoundary2D(df.SubDomain):
    """
    Trying to determine reference boundaries & reference points...
    Take the left and bottom edges as the reference, and
    remap the top and right edges onto them.
    """

    def __init__(self, Lx, Ly, **kwargs):
        df.SubDomain.__init__(self)
        self.Lx = Lx
        self.Ly = Ly
        # super().__init__(**kwargs)

        if verbose:
            print("[{}] PeriodicBoundary2D instantiated.".format(mpiRank))

    def inside(self, x, on_boundary):
        """
        Is the current point on the segments of the boundary
        we're using for reference values? Exclude the top and
        right segments.
        """
        # the boundary minus the top & right sides
        is_inside = bool(
            (df.near(x[0], 0) or df.near(x[1], 0))
            and (
                not (
                    (df.near(x[0], self.Lx) and df.near(x[1], self.Ly))
                    or (df.near(x[0], self.Lx) and df.near(x[1], self.Ly))
                )
            )
            and on_boundary
        )

        if verbose:
            print(
                "[{}] PeriodicBoundary2D::inside({}, {}) -> {}".format(
                    mpiRank, *x, is_inside
                )
            )

        return is_inside

    def map(self, x, y):
        # Map points from right boundary to left reference
        if df.near(x[0], self.Lx):
            y[0] = x[0] - self.Lx
        else:
            y[0] = x[0]

        # Map points from top boundary to bottom reference
        if df.near(x[1], self.Ly):
            y[1] = x[1] - self.Ly
        else:
            y[1] = x[1]

        if verbose:
            if (x[0] == y[0]) and (x[1] == y[1]):
                print("[{}] PeriodicBoundary2D::map(   {}, {})".format(mpiRank, *x))
            else:
                print(
                    "[{}] PeriodicBoundary2D::map(   {}, {}) -> ({}, {})".format(
                        mpiRank, *x, *y
                    )
                )


class PeriodicBoundary3D(df.SubDomain):
    def __init__(self, Lx, Ly, Lz, **kwargs):
        df.SubDomain.__init__(self)
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        super().__init__(**kwargs)

        if verbose:
            print("[{}] PeriodicBoundary3D instantiated.".format(mpiRank))

    def inside(self, x, on_boundary):
        # the boundary minus the top, right, and front sides
        return on_boundary and bool(
            (df.near(x[0], 0) or df.near(x[1], 0) or df.near(x[2], 0))
            and (
                not (
                    (
                        df.near(x[0], self.Lx)
                        and df.near(x[1], self.Ly)
                        and df.near(x[2], self.Lz)
                    )
                    or (
                        df.near(x[0], self.Lx)
                        and df.near(x[1], self.Ly)
                        and df.near(x[2], self.Lz)
                    )
                )
            )
        )

    def map(self, x, y):
        # Map points from right boundary to left reference
        if df.near(x[0], self.Lx):
            y[0] = x[0] - self.Lx
        else:
            y[0] = x[0]

        # Map points from top boundary to bottom reference
        if df.near(x[1], self.Ly):
            y[1] = x[1] - self.Ly
        else:
            y[1] = x[1]

        # Map points from front boundary to back reference
        if df.near(x[2], self.Lz):
            y[2] = x[2] - self.Lz
        else:
            y[2] = x[2]


class PeriodicDomain(df.SubDomain):
    def __init__(self, *lengths):
        """
        Initialize the periodic domain with the given dimensions.
        """
        self.lengths = lengths
        super().__init__()

    def inside(self, x, on_boundary):
        """
        Check if a point is inside the periodic domain.
        """
        inside = True
        for i, L in enumerate(self.lengths):
            inside = inside and (df.near(x[i], 0) or df.near(x[i], L))
        return inside

    def map(self, x, y):
        """
        Map a point from the periodic domain to the hypercube.
        """
        y[:] = x[:]
        for i, L in enumerate(self.lengths):
            if df.near(x[i], L):
                y[i] = x[i] - L


class PeriodicBoundary(df.SubDomain):
    def __init__(self, mesh, directions):
        """
        Initialize the periodic domain with the given dimensions.
        """
        super().__init__()
        self.L = np.max(mesh.coordinates(), axis=0)
        self.directions = np.asarray(directions)

    def inside(self, x, on_boundary):
        """
        Check if a point is inside the periodic domain.
        """
        # Returns True if the point is on the boundary
        return return df.near(x[0], 0) and on_boundary

    def map(self, x, y):
        """
        Map a point from the periodic domain to the hypercube.
        """
        # Apply periodic boundary conditions in specified directions
        y[:] = x
        for i, direction in enumerate(self.directions):
            if direction:
                L = self.L[i]
                y[i] = np.fmod(x[i] + L, L)

def set_functions(mesh, element, nprev, PB):

    if PB:
        pbc = PeriodicBoundary(mesh, PB) #maps left to right boundary
        V = df.FunctionSpace(mesh, element, constrained_domain=pbc)
    else:
        V = df.FunctionSpace(mesh, element)

    U = df.Function(V)

    U_n = []
    for i in range(nprev):
        U_n.append(df.Function(V))

    return U, U_n, V

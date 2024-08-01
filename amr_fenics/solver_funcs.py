# -----------------------------------------------------------------------------
# Copyright (c) 2022 Nana Ofori-Opoku. All rights reserved.

"""
**solver_funcs.py**
contains routines related problem definitions and solver operations.
"""

import dolfin as df
import amr_fenics.helpers as helpers
import amr_fenics.input_output as io


def set_backend(Domain, PetscOpt):
    # Form parameters for the backend

    df.parameters["std_out_all_processes"] = False
    df.parameters["form_compiler"]["optimize"] = True
    df.parameters["form_compiler"]["cpp_optimize"] = True
    df.parameters["form_compiler"]["representation"] = "uflacs"
    df.parameters["form_compiler"]["quadrature_degree"] = Domain["q"]
    df.parameters["form_compiler"]["precision"] = 300
    df.parameters["linear_algebra_backend"] = "PETSc"

    for i in PetscOpt:
        if PetscOpt[i] is None:
            df.PETScOptions.set(i)
        else:
            df.PETScOptions.set(i, PetscOpt[i])


class NonLinearProblem(df.NonlinearProblem):
    def __init__(self, a, L, bcs, nullsp=None):
        df.NonlinearProblem.__init__(self)
        self.bilinear_form = a
        self.linear_form = L
        self.bcs = bcs
        self.nullsp = nullsp

    def F(self, b, x):
        df.assemble(self.linear_form, tensor=b)
        [bc.apply(b, x) for bc in self.bcs]
        if self.nullsp is not None:
            self.nullsp.orthogonalize(b)

    def J(self, A, x):
        df.assemble(self.bilinear_form, tensor=A)
        [bc.apply(A) for bc in self.bcs]
        if self.nullsp is not None:
            df.as_backend_type(A).set_near_nullspace(self.nullsp)


def set_problem(F, U, V, bcs):
    dU = df.TrialFunction(V)
    J = df.derivative(F, U, dU)
    problem = NonLinearProblem(J, F, bcs)

    return problem


def set_solver(solverType, solverMethod, linSolver):
    # List of linear solvers
    # bicgstab       |  Biconjugate gradient stabilized method
    # cg             |  Conjugate gradient method
    # default        |  default linear solver
    # gmres          |  Generalized minimal residual method
    # minres         |  Minimal residual method
    # mumps          |  MUMPS (MUltifrontal Massively Parallel Sparse direct Solver)
    # petsc          |  PETSc built in LU solver
    # richardson     |  Richardson method
    # superlu        |  SuperLU
    # tfqmr          |  Transpose-free quasi-minimal residual method
    # umfpack        |  UMFPACK (Unsymmetric MultiFrontal sparse LU factorization)

    # List of preconditioners
    # amg              |  Algebraic multigrid
    # default          |  default preconditioner
    # hypre_amg        |  Hypre algebraic multigrid (BoomerAMG)
    # hypre_euclid     |  Hypre parallel incomplete LU factorization
    # hypre_parasails  |  Hypre parallel sparse approximate inverse
    # icc              |  Incomplete Cholesky factorization
    # ilu              |  Incomplete LU factorization
    # jacobi           |  Jacobi iteration
    # none             |  No preconditioner
    # petsc_amg        |  PETSc algebraic multigrid
    # sor              |  Successive over-relaxation

    if solverType == "newton":
        solver = df.NewtonSolver()
        prm = solver.parameters
        prm["linear_solver"] = linSolver
        # prm["preconditioner"] = 'ilu'
        prm["relative_tolerance"] = 1e-4
        prm["absolute_tolerance"] = 1e-8
        # prm["relaxation_parameter"] = .5
        prm["convergence_criterion"] = "incremental"  # residual, incremental
        prm["maximum_iterations"] = 20
        prm["error_on_nonconvergence"] = True

    # info(prm, True)
    if solverType == "snes":
        solver = df.PETScSNESSolver()
        prm = solver.parameters
        prm["method"] = solverMethod
        prm["linear_solver"] = linSolver

    # list_linear_solver_methods()
    return solver


def set_bdf1(u, u_n, v, dt, rhs, tau=None):
    if tau is None:
        return df.inner(u - u_n, v) * df.dx - dt * rhs
    else:
        return tau * df.inner(u - u_n, v) * df.dx - dt * rhs


def set_bdf2(u, u_n, u_nn, v, time, rhs, tau=None):
    dtn = time[0] - time[1]
    dtnn = time[1] - time[2]
    rn = dtn / dtnn

    if tau is None:
        return (
            df.inner(
                u
                - ((1 + rn) ** 2 / (1 + 2 * rn)) * u_n
                + (rn**2 / (1 + 2 * rn)) * u_nn,
                v,
            )
            * df.dx
            - (1 + rn) / (1 + 2 * rn) * dtn * rhs
        )
    else:
        return (
            tau
            * df.inner(
                u
                - u_n * (1 + rn) ** 2 / (1 + 2 * rn)
                + (rn**2 / (1 + 2 * rn)) * u_nn,
                v,
            )
            * df.dx
            - (1 + rn) / (1 + 2 * rn) * dtn * rhs
        )


def adjust_time_step(
    iter,
    min_iterations,
    max_iterations,
    amplification,
    reduction,
    dt,
    dt_max=None,
    adapt_count=None,
):
    if iter < min_iterations:
        if dt(0) == dt_max(0):
            dt.assign(dt_max(0))
        else:
            dt.assign(amplification * dt(0))
            if dt_max is not None:
                if dt(0) > dt_max(0):
                    dt.assign(dt_max(0))
            io.disp("Time step GROWTH  =====> dt = {}".format(dt(0)))

    if iter > max_iterations and (adapt_count is None):
        dt.assign(reduction * dt(0))
        io.disp("Time step DECLINE =====> dt = {}".format(dt(0)))

    if iter > max_iterations and (adapt_count is not None and adapt_count != 0):
        dt.assign(reduction * dt(0))
        io.disp("Time step DECLINE =====> dt = {}".format(dt(0)))

    return dt


def failed_solution_attempt(U, U_n, nprev, t, dt, iter, time_bdf=None):
    # Solution didn't converge...change decrease time step and attempt again
    io.disp(
        "Solution failed on iteration {0:2d} with time step dt {1:e}".format(
            iter, dt(0)
        )
    )
    t -= dt(0)  # Take off failed dt
    dt.assign(dt(0) * 0.5)  # assign new dt to be half
    io.disp("Time step DECLINE =====> dt = {}".format(dt(0)))
    t += dt(0)

    if time_bdf is not None:
        U.assign(U_n[0])
        for i in range(nprev - 1):
            U_n[i].assign(U_n[i + 1])

            for i in range(nprev):
                time_bdf[i].assign(time_bdf[i + 1](0))

        return U, U_n, t, time_bdf, dt

    else:
        U.assign(U_n[0])
        for i in range(nprev - 1):
            U_n[i].assign(U_n[i + 1])

        return U, U_n, t, dt

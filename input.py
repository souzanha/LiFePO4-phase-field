"""
Li intercalation Model by Souzan Hammadi & Nana Ofori-Opoku

Fields in F:

0. 𝔽: function space
1. 𝐹: Helmholtz free energy
2. x₁: composition of Li
3. |∇c|²: squared gradient of concentration
4. f(c): thermodynamic energy density

Read input parameters & settings from `config.py`
Output runtime values to `record.yml`
"""
from __future__ import print_function

# === Basic performance monitoring ===

import time

startTime = time.time()

# === System libraries ===
import gc
import numpy as np
import os
import pandas as pd
import ruamel.yaml

# import steppyngstounes as stp
import sys

# === Dolfin/FEniCS ===

import ufl
import dolfin as df

# === AMR and other secret sauce ===

import amr_fenics.input_output as io
import amr_fenics.mesh_domain_funcs as mdf
import amr_fenics.solver_funcs as slf
import amr_fenics.helpers as hlp
import amr_fenics.mesh_refinement as amr
import amr_fenics.utility as ut

import elchem_fenics.elchem as elchem

# === Job Parameters ==
from config import Adapt, Domain, Interval, Model, Solver, PetscOpt

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)

run = "." if len(sys.argv) < 2 else sys.argv[1]
restartFlag = os.path.isfile("{}/restart.h5".format(run)) #==True

if restartFlag:
    restart_run = sys.argv[2]
    fs = io.output_directory(restart_run)
    nrg_file = restart_run + "/" + "ene.csv"
    voltage_file = restart_run + "/" + "v_c.csv"
else:
    fs =  io.output_directory(run)
    nrg_file = run + "/" + "ene.csv"
    voltage_file = run + "/" + "v_c.csv"

names = [r"$c$", r"$\mu$", r"$u$"]
auxnames = [r"$\mathcal{F}$", r"$\mathcal{F}_{el}$", r"$\sigma$", r"$\Delta c$"]

record = {"data": {}, "domain": {}, "model": {}, "runtime": {}, "adapt": {}}

stats_head = [
    "time",  # "simulation time, dimensionless"
    "c",  # "Li composition, whole domain"
    "μ",  # average chemical potential
    "Helmholtz",  # "Helmholtz free energy, ∫F·dx"
    "wall_time",  # "runtime (sec)"
    "mem_GB",  # "current memory usage (GB)"
    "peak_GB",  # "maximum memory usage (GB)"
]

voltage_data = ["voltage", "conc", "mu_tot", "current", "overpotential"]

record["data"]["stats"] = {}
record["data"]["stats"]["file"] = nrg_file
record["data"]["stats"]["cols"] = stats_head

# === Runtime configuration ===

# Solver options
BDF2 = Solver["BDF2"]
dt = df.Constant(Interval["timestep"])  # time step
dt_max = Interval["maxtimestep"]  # maximum time step
t_interval_output = Interval["time_interval_output"]  # time interval output
cback = Interval["cutback"]  # reduction of time step
growth = Interval["growth"]  # growth of time step
min_its = Interval["min_iter"]  # minimum nonlinear iterations
max_its = Interval["max_iter"]  # maximum nonlinear iterations
checkpoint_frequency = Interval["f_checkpoint"]  # checkpoint frequency

D = record["model"]["DLi"] = float(Model["DLi"])  # diff coeff [m²/s]

σ = record["model"]["σ"] = Model["sigma"]  # surface energy
Wc = record["model"]["Wc"] = Model["Wc"]  # length scale

tc = record["model"]["tc"] = Wc**2 / D  # time scale
To = record["model"]["To"] = Model["To"]  # temperature
vm = record["model"]["vm"] = Model["vm"]  # molar volume
R = record["model"]["R"] = Model["R"]  # gas constant
RTv = R * To / vm
Om = record["model"]["Ω"] = Model["Omega"]  # enthalpy of mixing coeff

Δf = -RTv * df.ln(2) + Om / vm / 4  # energy barrier scale [J/m³]

H = record["model"]["H"] = σ / Wc  # energy scale

Δϕ = record["model"]["Δϕ"] = Model["Δϕ"]  # [V] overpotential
μeq = record["model"]["μeq"] = Model["μeq"]  # [J/m³] equilm chemical potential for To
λᵣ = record["model"]["λᵣ"] = Model["λᵣ"]  # [J] reorganizational energy for the MHC flux

flux = record["model"]["flux"] = Model["flux"]  # surface flux type, i.e. kinetics

j0 = Model["j0"]

if j0 is None:
    k0 = record["model"]["k0"] = Model["k0"]  # rate constant
    j0 = record["model"]["j0"] = (Model["F"] / (Model["Nₐ"] *  Model["Wc"]**2)) * Model["k0"]
else:
    j0 = record["model"]["j0"] = Model["j0"]

j0coeff = Model["vm"] * j0 / Model["F"]  # const coeff for BV flux B.C
μelec = Model["μeq"] - (
    Model["F"] * Model["Δϕ"] / Model["vm"]
)  # electrolyte chemical potential

# MHC B.C. parameter, i.e. reorganization energy and adjustment constant
λᵣ = record["model"]["λᵣ"] = Model["λᵣ"]

Const = Model["Const"]

ν = Model["ν"]  # possion's ration
E = Model["E"]  # Young's Modulus
C11 = (1 - ν) / ν
C12 = 1
C44 = (1 - 2 * ν) / (2 * ν)

# First Lamé constant to convert modulus and poisson ratio to stiffness
λ = (E * ν) / ((1 + ν) * (1 - 2 * ν))  # Lame's Constant
μ = E / (2 * (1 + ν))  # Shear modulus or Second Lame Constant

# Eigen strains
ϵ11 = Model["e11"]
ϵ22 = Model["e22"]
ϵ33 = Model["e33"]

# === Non-dimensionalize model parameters ===
T = record["runtime"]["tfinal"] = Interval["runtime"] / tc  # total time (max)

io.disp("Critical & controlling parameters:")
io.disp(f"     Δf={Δf:<12g}    H={H:<12g}")
io.disp(f"     D={D:<12g}")
io.disp(f"     t={tc:<12g}     T={T:<12g}")

# === Define the mesh, size & resolution ===
bcs = None
PB = Domain["PB"]  # [True, False]

"""
The program starts with a coarse mesh, and refines it — dividing the
element size by two — until the target element size is reached.
A heuristic sense is that 5 to 8 levels of refinement delivers a
good mesh, meaning the starting point (emax) depends on the goal (nde):

    emax = nde * 2ʳ⁻¹ = 64 * nde where r is the resolution in this case 7
"""
emax = record["domain"]["emax"] = Domain["emax"]
Lx = record["domain"]["Lx"] = Domain["Lx"]  # domain length
Ly = record["domain"]["Ly"] = Domain["Ly"]  # domain width
Lz = record["domain"]["Lz"] = Domain["Lz"]  # domain width

Nx = record["domain"]["Nx"] = int(Lx / emax)  # number of initial elements x-dir
Ny = record["domain"]["Ny"] = int(Ly / emax)  # number of initial elements y-dir

if Lz is not None:
    Nz = record["domain"]["Ny"] = int(Lz / emax)  # number of initial elements y-dir

emin = record["domain"]["emin"] = Domain["nde"]

io.disp(f"     smallest element, cartesian distance(i.e. Δx)={emin}")

# dimensionless entropy of mixing coefficient
RTv = record["model"]["RTv"] = RTv / H

# dimensionless enthalphy of mixing
Om = Om / vm / H

# Diffusion coefficient
Dm = record["model"]["Dm"] = D * (tc / Wc**2) / RTv  # dimensionless

# dimensionless BV B.C. parameters
j0coeff = record["model"]["j0coeff"] = j0coeff * tc / Wc
μelec = record["model"]["μelec"] = μelec / H

# dimensionless MHC B.C. parameter
#λᵣ = record["model"]["λ̄ᵣ"] = λᵣ * Model["Nₐ"] / vm / H

VHf = vm * H / Model["F"]  # V_m*H/F

io.disp(f"     RTv={RTv:<12g}   μelec={μelec:<12g}")
io.disp(f"     Dm={Dm:<11g}     j0coeff={j0coeff:<12g}")

# Create four free indices for defining the scalar product
i, j, k, l = ufl.indices(4)

if Lz is None:
    io.disp("\n ********** This Simulation will be in 2D ********** \n")
    C = (
        df.Constant(1)
        * (λ / H)
        * ufl.as_tensor([[C11, C12, 0], [C12, C11, 0], [0, 0, C44]])
    )
    ϵ0 = ufl.as_tensor([[ϵ11, 0], [0, ϵ22]])
else:
    io.disp(" ********** This Simulation will be in 3D ********** ")
    C = (λ / H) * ufl.as_tensor(
        [
            [C11, C12, C12, 0, 0, 0],
            [C12, C11, C12, 0, 0, 0],
            [C12, C12, C11, 0, 0, 0],
            [0, 0, 0, C44, 0, 0],
            [0, 0, 0, 0, C44, 0],
            [0, 0, 0, 0, 0, C44],
        ]
    )
    ϵ0 = ufl.as_tensor([[ϵ11, 0, 0], [0, ϵ22, 0], [0, 0, ϵ33]])

# setting some backend parameters and PETSc solver options directions
slf.set_backend(Domain, PetscOpt)


# Class representing the intial conditions
class InitialConditions(df.UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, values, x):
        if Δϕ > 0.0:  # delthitation
            values[0] = 1.0  # concentration
        else:  # lithiation
            values[0] = 0.0
        values[1] = 0.0  # chemical potential
        values[2] = 0.0  # displacement in x-dir
        values[3] = 0.0  # displacement in y-dir
        if Lz is not None:
            values[4] = 0.0  # displacement in z-dir

    def value_shape(self):
        if Lz is not None:
            return (5,)
        else:
            return(4,)


def voigt(u):
    if u.ufl_shape == (2, 2):
        return ufl.as_vector([u[i, i] for i in range(2)] + [2 * u[0, 1]])
    if u.ufl_shape == (3, 3):
        return ufl.as_vector(
            [u[i, i] for i in range(3)] + [2 * u[1, 2]] + [2 * u[0, 2]] + [2 * u[0, 1]]
        )


def strain(u):
    return (1 / 2) * (df.grad(u) + df.grad(u).T)


def sigma(C, ϵ):
    if ϵ.ufl_shape == (2, 2):
        σ = ufl.as_tensor([C[i, j] * ϵ[j]], (i, j))
    if ϵ.ufl_shape == (3, 3):
        σ = ufl.as_tensor([C[i, j, k, l] * ϵ[j, l]], (i, j))

    return σ


def f_elas(C, u, c, ϵ0):
    ϵ = strain(u)
    ϵtot = ϵ - ϵ0 * c

    σ = C * voigt(ϵtot)
    fel = (1 / 2) * df.inner(σ, voigt(ϵtot))

    return σ, fel


def plog(c, tol):
    Δc = c - tol

    t0 = df.ln(tol)
    t1 = Δc / tol
    t2 = -(Δc**2) / (2 * tol**2)
    t3 = Δc**3 / (3 * tol**3)

    return ufl.conditional(ufl.lt(c, tol), t0 + t1 + t2 + t3, df.ln(c))


def f_chem(rtv, om, c):
    entropy = rtv * (c * plog(c, 1e-3) + (1 - c) * plog(1 - c, 1e-3))
    enthalpy = om * c * (1 - c)

    return entropy + enthalpy


def equation_solve(U, U0, ME, mesh, element, dt, BDF2, time_bdf2):
    # Split mixed functions
    c, mu, u = df.split(U)
    c0, mu0, u0 = df.split(U0[0])

    if BDF2:
        c00, mu00, u00 = df.split(U0[1])

    q, v, w = df.TestFunctions(ME)

    nME = df.FunctionSpace(mesh, element)
    cv = df.Function(nME)
    c_v = df.variable(cv)

    replacement = {cv: c}

    fdens = f_chem(RTv, Om, c_v)
    σ, fel = f_elas(C, u, c_v, ϵ0)
    fgrad = (1 / 2) * df.inner(df.grad(c_v), df.grad(c_v))
    Func = fgrad + fdens + fel

    μₘ = (1.0 - Solver["theta"]) * mu0 + Solver["theta"] * mu  # mu_(n+theta)

    if flux == "BV":
        # Butler-Volmer flux
        JLi = elchem.Butler_Volmer(mu, j0coeff, μelec, RTv)
    else:
        # Marcus-Hush-Chidsey flux
        JLi = elchem.MHC(λᵣ, mu, μelec, RTv, j0coeff, Const)

    # Weak statement of the equations

    # Linear equation for c -> ∫ ∂c/∂t vₜₑₛₜ  = D(c)∇μ⋅∇vₜₑₛₜ + JBV⋅n̂ vₜₑₛₜ
    rhs_c = -Dm * c * (1 - c) * df.inner(df.grad(μₘ), df.grad(q)) * df.dx
    rhs_c += JLi * q * ds

    if BDF2:
        L_c = slf.set_bdf2(c, c0, c00, q, time_bdf2, rhs_c)
    else:
        L_c = slf.set_bdf1(c, c0, q, dt, rhs_c)

    # Linear equation for mu -> ∫ μ*vₜₑₛₜ = δF/δc⋅vₜₑₛₜ
    dF = df.derivative(Func, cv, v)
    rhs_mu = ufl.replace(dF, replacement) * df.dx

    L_mu = mu * v * df.dx - rhs_mu

    # Linear equation for u (displacements) ->  ∫ σ⋅ϵₜₑₛₜ
    ϵtest = strain(w)
    rhs_u = df.inner(ufl.replace(σ, replacement), voigt(ϵtest)) * df.dx
    L_el = rhs_u

    L = L_c + L_mu + L_el

    fgradc = df.sqrt(df.inner(df.grad(c), df.grad(c)))

    Func = ufl.replace(Func, replacement)

    return L, Func, fel, σ, fgradc 


def update_voltage_profile(
    U, nmesh, j0coeff, μelec, RTv, vprofile, λᵣ, Const, flux, Δϕ, VHf, ds, P1
):
    conc, mu_tot, voltage, current = elchem.voltage_profile(
        U, nmesh, j0coeff, μelec, RTv, λᵣ, Const, flux, Δϕ, VHf, ds, P1
    )
    if hlp.isHead:
        index = [0] if vprofile is None else [len(vprofile)]
        values = np.array([[voltage, conc, mu_tot, current, Δϕ]])

        update = pd.DataFrame(values, columns=voltage_data, index=index)

        if vprofile is None:
            return update
        else:
            return pd.concat([vprofile, update])


def boundaries(mesh, V, bcs):
    # BOUNDARY CONDITIONS
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    # 100 -  surfaces
    left = mdf.xboundary(0)
    right = mdf.xboundary(Lx)

    left.mark(boundaries, 1)
    right.mark(boundaries, 2)

    # 010 - surfaces
    bottom = mdf.yboundary(0)
    top = mdf.yboundary(Ly)

    bottom.mark(boundaries, 3)
    top.mark(boundaries, 4)

    # 001 - surfaces
    if Lz is not None:
        back = mdf.zboundary(0)
        front = mdf.zboundary(Lz)

        back.mark(boundaries, 5)
        front.mark(boundaries, 6)

    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Traction-Free boundary conditions for y surfaces
    bcs.append(df.DirichletBC(V.sub(2).sub(1), df.Constant((0)), bottom))
    bcs.append(df.DirichletBC(V.sub(2).sub(1), df.Constant((0)), top))

    if Lz is None:
        # Zero-Displacement x surfaces
        bcs.append(df.DirichletBC(V.sub(2), df.Constant((0, 0)), left))
        bcs.append(df.DirichletBC(V.sub(2), df.Constant((0, 0)), right))
    else:
        # Zero-Displacement x surfaces
        bcs.append(df.DirichletBC(V.sub(2), df.Constant((0, 0, 0)), left))
        bcs.append(df.DirichletBC(V.sub(2), df.Constant((0, 0, 0)), right))

        # Zero-Displacement z surfaces
        bcs.append(df.DirichletBC(V.sub(2), df.Constant((0, 0, 0)), back))
        bcs.append(df.DirichletBC(V.sub(2), df.Constant((0, 0, 0)), front))

    return bcs, ds(3)


def update_stats(t, F, U, stats_df):
    c, mu, u = df.split(U)

    cavg = ut.integrate_func(c) / msize  # Li average
    μavg = ut.integrate_func(mu) / msize  # chemical potential average
    Favg = ut.integrate_func(F[0]) / msize  # Helmholtz free energy

    memUsed, memPeak = ut.memory_usage(as_string=False)

    if hlp.isHead:
        index = [0] if stats_df is None else [len(stats_df)]
        timer = time.time() - startTime
        values = np.array([[t * tc, cavg, μavg, Favg, timer, memUsed, memPeak]])
        update = pd.DataFrame(values, columns=stats_head, index=index)

        if stats_df is None:
            return update
        else:
            return pd.concat([stats_df, update])


def output_stats(nrg_file, stats_df, vprofile, vfile, report=False):
    if hlp.isHead:
        stats_df.to_csv(nrg_file, sep="\t", index=False)
        vprofile.to_csv(vfile, sep="\t")
        if report:
            latest = stats_df.iloc[-1]
            io.disp("├─────────────────────────────────┐")
            io.disp("│ comp Li      = {:<10.6f}".format(latest["c"]))
            io.disp("│ chem pot μ   = {:<10.6f}".format(latest["μ"]))
            io.disp("│ enrg 𝓕       = {:<-10.5f}".format(latest["Helmholtz"]))
            io.disp("├─────────────────────────────────┘")


def output_data(t, F, U):
    io.write_fields(U, t, names, fs)

    for n, f in zip(auxnames, F):
        if n == "$\sigma$":
            if Lz is not None:
                io.write_aux(
                    f, t, n, df.TensorFunctionSpace(nmesh, "DG", degree=0, shape=(6,)), fs)
            else:
                io.write_aux(
                    f, t, n, df.TensorFunctionSpace(nmesh, "DG", degree=0, shape=(3,)), fs)
        else:
            io.write_aux(f, t, n, df.FunctionSpace(nmesh, P1), fs)

if BDF2:
    nprev = 2
    time_bdf2 = [df.Constant(0), df.Constant(0), df.Constant(0)]
else:
    nprev = 1
    time_bdf2 = None

initStart = time.time()

io.disp("┌─ Setting up simulation.")
# Create mesh and build function space
# Initial mesh

if Lz is None:
    imesh = df.RectangleMesh.create(
        [df.Point(0, 0), df.Point(Lx, Ly)],
        [Nx, Ny],
        df.CellType.Type.triangle,
        Domain["diag"],
    )
    orgmesh = df.RectangleMesh.create(
        [df.Point(0, 0), df.Point(Lx, Ly)],
        [Nx, Ny],
        df.CellType.Type.triangle,
        Domain["diag"],
    )

else:
    imesh = df.BoxMesh.create(
        [df.Point(0, 0, 0), df.Point(Lx, Ly, Lz)],
        [Nx, Ny, Nz],
        df.CellType.Type.tetrahedron,
    )
    orgmesh = df.BoxMesh.create(
        [df.Point(0, 0, 0), df.Point(Lx, Ly, Lz)],
        [Nx, Ny, Nz],
        df.CellType.Type.tetrahedron,
    )

if restartFlag:

    P1, element, ME, nmesh, U, U0, t, dt = io.read_restart(run, dt, nprev, PB, orgmesh)

    resolution = amr.calc_resolution(emax, emin)
    imesh = amr.initial_mesh_elect(orgmesh, resolution, emax)

    vprofile = None
    stats_df = None

    U0[0].assign(U)

else:

    P1 = df.FiniteElement("Lagrange", imesh.ufl_cell(), 1)
    Pv = df.VectorElement("Lagrange", imesh.ufl_cell(), 1)
    element = df.MixedElement([P1, P1, Pv])

    resolution = amr.calc_resolution(emax, emin)
    imesh = amr.initial_mesh_elect(imesh, resolution, emax) #=imesh

    nmesh = df.Mesh(imesh)

    U, U0, ME = mdf.set_functions(nmesh, element, nprev, PB)
    
    # Create intial conditions and interpolate
    U_init = InitialConditions(degree=1)
    U.interpolate(U_init) #error here
    U0[0].assign(U)
    
bcs = []
bcs, ds = boundaries(nmesh, ME, bcs)
# Create nonlinear problem and solver
L, *F = equation_solve(U, U0, ME, nmesh, P1, dt, BDF2, time_bdf2)
problem = slf.set_problem(L, U, ME, bcs)
solver = df.PETScSNESSolver()

msize = df.assemble(
    df.interpolate(df.Constant(1.0), df.FunctionSpace(nmesh, P1)) * df.dx
)

# Step in time

if not restartFlag:
    t = 0.0

    c_init = df.assemble(U[0] * df.dx)

    stats_df = update_stats(t, F, U, None)
    vprofile = update_voltage_profile(
        U, nmesh, j0coeff, μelec, RTv, None, λᵣ, Const, flux, Δϕ, VHf, ds, P1
    )
    output_stats(nrg_file, stats_df, vprofile, voltage_file, report=True)

    output_data(t, F, U)
    io.disp("├─ Saving initial state:", end=" \n")

    # Write summary record to YAML
    if hlp.isHead:
        with open("{}/record.yml".format(run), "w") as fh:
            yml = ruamel.yaml.YAML(typ=["unsafe", "pytypes"])
            yml.allow_unicode = True
            yml.default_flow_style = False
            yml.explicit_start = True
            yml.indent(mapping=2, sequence=4, offset=0)
            yml.dump(record, fh)

# iteraction counter for converged solutions
iter_s = 1

if BDF2:
    # generate second condition and redefine problem
    io.disp("├─ Step 1…")
    io.disp("├─── BDF2 secondary condition …")
    eulerStart = time.time()

    L, *F = equation_solve(U, U0, ME, nmesh, P1, dt, False, time_bdf2)
    problem = slf.set_problem(L, U, ME, bcs)

    BDF2_ini_count = 0
    converged = False

    while not converged:
        BDF2_ini_count += 1
        try:
            U0[0].assign(U)
            nits, converged = solver.solve(problem, U.vector())
        except RuntimeError:
            converged = False

        if converged:
            t += dt(0)
            iter_s += 1
            time_bdf2[1].assign(time_bdf2[0](0))
            time_bdf2[0].assign(t)
        else:
            dt.assign(dt(0) / 2)  # assign new dt to be half
            io.disp("Time step DECLINE ===> dt={}".format(dt(0)))
            U.assign(U0[0])

            if BDF2_ini_count > 10:
                io.disp("Failed 10× to calculate secondary BDF2 condition")
                io.disp("==== Terminating Simulation ====")
                hlp.mpiComm.Abort()

    # Re-define problem and form
    L, *F = equation_solve(U, U0, ME, nmesh, P1, dt, BDF2, time_bdf2)
    problem = slf.set_problem(L, U, ME, bcs)

    output_data(t, F, U)

    grammar = "try" if BDF2_ini_count == 1 else "tries"
    timer = time.time() - eulerStart

    io.disp(f"├── {BDF2_ini_count} {grammar} ({nits} iters) in {timer:6.0f} s.")

stats_df = update_stats(t, F, U, None)

vprofile = update_voltage_profile(
    U, nmesh, j0coeff, μelec, RTv, vprofile, λᵣ, Const, flux, Δϕ, VHf, ds, P1
)

output_stats(nrg_file, stats_df, vprofile, voltage_file, report=True)
timer = time.time() - initStart
io.disp(f"└─ System Initialized in {timer:.0f} s\n")

# === Evolve the System Microstructure ===

time_to_refine = Adapt["remesh"]
time_to_refine -= 1
time_to_print = Interval["domain"]
time_to_print -= 1
amrStart = Adapt["amrStart"]

adaptTime = 0

# flag for hysteresis cycle (both lithiation and delithiation occuring)
hystCycle = True

# flag for simulation continuance
goSim = True

io.disp(f"┌─ Starting the system evolution of microstructure.")
io.disp(f"│")

flag_interval_output = 0

#while t < T:
while goSim:
    try:
        step = dt(0)

        if time_to_refine == Adapt["remesh"]:
            leaf = "┌"
        elif time_to_refine == 1:
            leaf = "└"
        else:
            leaf = "├"

        io.disp(
            f"│  {leaf}─ Step {iter_s:<7d}| 𝑡->{t:>10f} / {T:<10f} with Δ𝑡={step:<6f}…"
        )
        t += step

        stepStart = time.time()

        for i in range(nprev - 1, 0, -1):
            U0[i].assign(U0[i - 1])

        U0[0].assign(U)

        if BDF2:
            for i in range(nprev, 0, -1):
                time_bdf2[i].assign(time_bdf2[i - 1](0))
            time_bdf2[0].assign(t)

        nits, converged = solver.solve(problem, U.vector())
        residual = solver.snes().norm
        iter_s += 1

        stats_df = update_stats(t, F, U, stats_df)
        vprofile = update_voltage_profile(
            U, nmesh, j0coeff, μelec, RTv, vprofile, λᵣ, Const, flux, Δϕ, VHf, ds, P1
        )

        timer = time.time() - stepStart
        io.disp(
            f"│  ├───── {timer:6.2f} s: {nits:>2d} iterations, residual={residual::<12g} "
        )

        time_to_print -= 1
        if time_to_print == 0 or flag_interval_output == 1:
            time_to_print = Interval["domain"]

            output_data(t, F, U)
            output_stats(nrg_file, stats_df, vprofile, voltage_file, report=True)
            if flag_interval_output == 1:

                io.disp(f"│ ├─ Writing time interval output"
                )

            flag_interval_output = 0

        # Iteration adaptive timestepper
        if nits == min_its:
            dt_checkpoint = abs(t%-t_interval_output)
            if dt_checkpoint > 0 and dt_checkpoint < dt(0) and dt_checkpoint < dt_max:
                dt.assign(dt_checkpoint)
                flag_interval_output = 1

        if nits < min_its:
            dt_checkpoint = abs(t%-t_interval_output)
            if dt_checkpoint  > 0 and dt_checkpoint < (dt(0)*growth) and dt_checkpoint < dt_max:
                dt.assign(dt_checkpoint)
                flag_interval_output = 1
            else:
                dt.assign(dt(0) * growth)
                if dt(0) >= dt_max:
                    dt.assign(dt_max)

        elif nits > max_its:
            dt_checkpoint = abs(t%-t_interval_output)
            if dt_checkpoint > 0 and dt_checkpoint < (dt(0)*cback) and dt_checkpoint < dt_max:
                dt.assign(dt_checkpoint)
                flag_interval_output = 1
            else:
                dt.assign(dt(0) * cback)

        if iter_s % checkpoint_frequency == 0:
            io.disp(f"│ ├─ Writing checkpoint")
            io.output_restart(run, imesh, nmesh, U, t, dt)

        if iter_s > amrStart or restartFlag == True:
            time_to_refine -= 1
            if time_to_refine == 0:
                time_to_refine = Adapt["remesh"]

                df.MPI.barrier(hlp.mpiComm)
                io.disp(f"│ ")
                io.disp(f"│ ┌─      *** Starting remeshing cycle ***")

                adaptStart = time.time()

                Ur = U.copy(deepcopy=True)
                if BDF2:
                    Ur0 = U0[0].copy(deepcopy=True)

                io.disp(f"│ ├─      Current solution copied")

                nmesh = amr.refine_coarsen(
                    imesh, nmesh, P1, resolution, Ur, Adapt["tol"], Adapt["beta"]
                )

                # Define functions
                U, U0, ME = mdf.set_functions(nmesh, element, nprev, PB)

                io.disp(f"│ ├─      Function space and solution vectors re-defined")

                df.LagrangeInterpolator.interpolate(U, Ur)
                if BDF2:
                    df.LagrangeInterpolator.interpolate(U0[0], Ur0)

                io.disp(f"│ ├─      Projection complete")

                bcs = []
                bcs, ds = boundaries(nmesh, ME, bcs)
                L, *F = equation_solve(U, U0, ME, nmesh, P1, dt, BDF2, time_bdf2)
                problem = slf.set_problem(L, U, ME, bcs)
                solver = df.PETScSNESSolver()

                timer = time.time() - adaptStart
                io.disp(f"│ ├─      Problem redefined")
                io.disp(f"│ ├─      Refinement cycle took {timer:5.2g} s")
                io.disp(f"│ └─      *** Finished remeshing cycle ***")
                io.disp(f"│ ")
                df.MPI.barrier(hlp.mpiComm)

                adaptTime += timer

        #equilμ = ut.equil_gradient(U, nmesh, P1, 1e-5, 1)

        #μavg_diff=(ut.integrate_func(df.split(U0[0])[1]) - ut.integrate_func(df.split(U)[1])) / msize

        cavg = ut.integrate_func(df.split(U)[0]) /msize

        #if equilμ:
        #if abs(μavg_diff) <= 1e-15:
        if cavg < 0.0002:
            if not hystCycle:
                io.disp("├──────────────────────────────────┐")
                io.disp("│    First equilirbium reached     │")
                io.disp("│    Reversing Cycle               │")
                io.disp("├──────────────────────────────────┘")

                hystCycle = True

                # adjust potential
                Δϕ = -1 * Model["Δϕ"]  # [V] overpotential
                μeq = Model["μeq"]  # [J/m³] equilm chemical potential for To
                μelec = μeq - (Model["F"] * Δϕ / vm)  # electrolyte chemical potential

                # dimensionless BV B.C. parameters
                μelec /= H

                iter_s = 0
                # amrStart *= 10
                time_to_refine = Adapt["remesh"]

                Ur = U.copy(deepcopy=True)
                if BDF2:
                    Ur0 = U0[0].copy(deepcopy=True)

                io.disp(f"│ ├─ Current solution copied")

                nmesh = amr.initial_mesh_elect(imesh, resolution, emax)

                # Define functions
                U, U0, ME = mdf.set_functions(nmesh, element, nprev, PB)

                io.disp(f"│ ├─ Function space and solution vectors re-defined")

                df.LagrangeInterpolator.interpolate(U, Ur)
                if BDF2:
                    df.LagrangeInterpolator.interpolate(U0[0], Ur0)

                io.disp(f"│ ├─ Projection complete")

                bcs = []
                bcs, ds = boundaries(nmesh, ME, bcs)
                L, *F = equation_solve(U, U0, ME, nmesh, P1, dt, BDF2, time_bdf2)
                problem = slf.set_problem(L, U, ME, bcs)
                solver = df.PETScSNESSolver()
            else:
                # if iter_s > amrStart:
                goSim = False
                io.disp("├──────────────────────────────────┐")
                io.disp("│   Second equilirbium reached     │")
                io.disp("│   Hystersis complete. Exiting!   │")
                io.disp("├──────────────────────────────────┘")

        gc.collect()
    except:
        t -= dt(0)

        io.disp("├─ Failure. {} its, 𝑡-={}".format(nits, dt(0)))

        dt.assign(dt(0) * cback)

        U.vector()[:] = U0[0].vector()
        for i in range(nprev - 1):
            U0[i].vector()[:] = U0[i + 1].vector()

        if BDF2:
            for i in range(nprev):
                time_bdf2[i].assign(time_bdf2[i + 1](0))

timer = time.time() - startTime
time_h, time_s = divmod(int(timer), int(3600))
time_m, time_s = divmod(int(time_s), int(60))

adapttime_h, adapttime_s = divmod(int(adaptTime), int(3600))
adapttime_m, adapttime_s = divmod(int(adapttime_s), int(60))

io.disp(f"│")
io.disp(
    f"├─ Total time spent remeshing {adaptTime:.2f} s or ({adapttime_h:2d}:{adapttime_m:02d}:{adapttime_s:02d})"
)
io.disp(
    f"└─ Simulation took {timer:.2f} s ({time_h:2d}:{time_m:02d}:{time_s:02d}) to reach {t:.2f}"
)
# List timings; average across processes in parallel
df.list_timings(df.TimingClear.keep, [df.TimingType.wall])

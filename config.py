"""
Input file for Li-intercalation simulations
Questions/comments to Souzan Hammadi

This file is Python dictionary, serving as both metadata and simulation
parameter input at runtime. The script also writes its runtime metadata as
record.yml, in YAML format rather than a Python dict.
"""

Repository = {
    "url": "https://github.com/souzanha/fenics-crabcakes",
    "branch": "BBB",  # git rev-parse --abbrev-ref HEAD
    "commit": "MMM",  # git rev-parse HEAD
    "pyhash": "HHH",  # sha1sum ternary_3phase_trevor.py | awk '{print $1}'
    "date": "DATE",
}

Adapt = {
    "remesh": 25,  # timesteps between refinement attempts
    "amrStart": 400,  # what iteration count to begin remeshing
    "beta": [  # weights for adaptive meshing gradients
        1.0,  # concentration (c)
        0.0,  # chemical potential (μ)
    ],
    "tol": 5e-3, # gradient tolerance for remeshing
}

Domain = {
    "Lx": 64,  # [Wc], domain length
    "Ly": 32,  # [Wc], domain width
    "Lz": None,  # [Wc], domain width, if None -> 2D
    "nde": 0.25,  # [Wc], minimum orthogonal element size
    "emax": 4,  # maximum size of initial coarse element
    "p": 1,  # element/polynomial degree (higher -> more coefficients)
    "q": 2,  # quadrature_degree (higher -> more nodes in element)
    "diag": "right/left",  # finite element node topology
    "PB": [True, False, False],  # periodic BCs
}

Interval = {
    "timestep": 1e-3,  # initial dt
    "restart": False,  # write restart-checkpoints?
    "f_checkpoint": 100,  # checkpoint frequency
    "time_interval_output": 100,  # time interval output
    "energy": 5,  # energy summary
    "domain": 100,  # XDMF checkpoint
    "runtime": 1e-3,  # seconds
    "cutback": 0.5,  # how much to reduce timestep
    "growth": 1.5,  # how much to increase timestep
    "min_iter": 3,  # minimum number of nonlinear iterations
    "max_iter": 8,  # maximum number of nonlinear iterations
    "maxtimestep": 1000,  # maximum timestep
}

Model = {
    "flux": "BV",  # BV:Butler-Volmer, MHC: Marcus-Hush-Chidsey
    "k0": 2.035e-4,  # [s^-1] rate constant
    "j0": None,  # [A/m²] exchange current density
    "Δϕ": 100e-3,  # [V] overpotential, positive=delithiation c_0=1
    "Wc": 1e-9,  # [m], interface width & The One True Length
    "sigma": 0.072,  # [J/m²], surface energy  ( 0.96 J/m² from Anton's paper)
    "DLi": 1e-15,  # [m²/s], isotropic Li diffusion for 3% antisites
    "Omega": 12e3,  # [J/mol] enthalpy of mixing coefficient
    "R": 8.3145,  # [J/mol/K] gas constant
    "To": 298,  # [K] temperature
    "vm": 4.38e-5,  # [m³/mol] molar volume
    "F": 96485.33,  # [C/mol] Faraday's constant
    "Nₐ": 6.0221408e23,  # [1/mol] Avogadro's number
    "λᵣ": 8.3,  # reorganizational energy for the MHC flux, scaled to KbT
    "Const": 3.358,  # constant that adjusts the MHC [ln(J)] to the BV [ln(J)] at low overpotentials
    "μeq": -0.0000211,  # [J/m³] equilm chemical potential for To
    "E": 125.7e9,  # [Pa] youngs modulus
    "ν": 0.252,  # [] poisson ratio
    "e11": 0.05,  #0.05 [] lattice mismatch along [100]
    "e22": 0.036,  #0.036 [] lattice mismatch along [010]
    "e33": -0.02,  #-0.02 [] lattice mismatch along [001]
}

Solver = {
    "BDF2": True,  # increase solve time & stability, decrease error
    "theta": 0.5,  # Crank-Nicolson parameter (fully implicit when 𝜃=1)
    "type": "snes",
    "method": "newtonls",
    "linear": "gmres",
}

PetscOpt = {
    # === PETSc Options ===
    # Scalable Nonlinear Equations Solvers (SNES)
    # <https://petsc.org/release/docs/manual/snes/>
    "snes_monitor_cancel": None,
    # "snes_converged_reason": None,
    "snes_rtol": 1e-6,
    "snes_atol": 1e-8,
    "snes_stol": 1e-12,
    # basic, bt, l2, cp, nleqerr, shell
    "snes_linesearch_type": "basic",
    "snes_max_it": 10,
    # Krylov Subspace Methods (KSP)
    # <https://petsc.org/release/docs/manual/ksp/>
    "ksp_rtol": 1e-10,
    "ksp_atol": 1e-14,
    "ksp_max_it": 1000,
    "sub_ksp_type": "preonly",
    "ksp_gmres_restart": 31,
    "ksp_gmres_cgs_refinement_type": "refine_ifneeded",
    # "ksp_view'": None,
    # 'ksp_monitor_true_residual')
    # 'ksp_converged_reason')
    # Preconditioner (PC)
    # <https://petsc.org/release/docs/manual/ksp/#preconditioning-within-ksp>
    "pc_type": "asm",  # jacobi sor
    # [basic,restrict,interpolate,none]
    "pc_asm_type": "interpolate",
    "pc_asm_overlap": 2,
    "sub_pc_type": "lu",
    # 'pc_factor_levels', '10')
}

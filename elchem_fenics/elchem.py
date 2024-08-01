# -----------------------------------------------------------------------------
# Copyright (c) 2023 Souzan Hammadi. All rights reserved.

"""
**elchem.py**
Flux equations for Butler-Volmer and Marcus-Hush-Chidsey kinetics
Calculation of the electrode potential
OCV = V_LFP - V_Li
"""

import numpy as np
import dolfin as df
import ufl

import math


def MHC(λ, μ, μelec, RTv, j0coeff, Const):  # Marcus-Hush-Chidsey
    Δμ = (μelec - μ) / (RTv)

    exp = (λ - ufl.sqrt(1 + ufl.sqrt(λ) + Δμ**2 )) / (2 * ufl.sqrt(λ))

    A = Const

    J = A * (j0coeff * df.sqrt(math.pi * λ) * ufl.tanh(Δμ / 2) * (1 - ufl.erf(exp)))

    return J


def Butler_Volmer(μ, j0coeff, μelec, RTv):  # Butler-Volmrr
    Δμ = (μelec - μ) / (2 * RTv)
    J = j0coeff * (df.exp(Δμ) - df.exp(-Δμ))
    return J


def calc_voltage(μtot, Δϕ, VHf):
    voltage = ((-1 * μtot) * VHf) - Δϕ
    return voltage


def voltage_profile(u, mesh, j0coeff, μelec, RTv, λ, Const, flux, Δϕ, VHf, ds, P1):
    msize = df.assemble(
        df.interpolate(df.Constant(1.0), df.FunctionSpace(mesh, P1)) * df.dx
    )
    lsize = df.assemble(
        df.interpolate(df.Constant(1.0), df.FunctionSpace(mesh, P1)) * ds
    )

    if flux == "BV":
        J = Butler_Volmer(u[1], j0coeff, μelec, RTv) * ds
    else:
        J = MHC(λ, u[1], μelec, RTv, j0coeff, Const) * ds

    I = df.assemble(J) / lsize  # current

    conc = df.assemble(u[0] * df.dx) / msize

    μtot = df.assemble(u[1] * df.dx) / msize

    voltage = calc_voltage(μtot, Δϕ, VHf)

    return conc, μtot, voltage, I

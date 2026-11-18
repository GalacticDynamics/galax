"""Experimental orbit integration."""

__all__ = ["SymplecticSolverT", "integrate_orbit"]

from typing import TypeAlias

import diffrax as dfx

import galax.fields as gf
from .integrate import integrate_orbit

SymplecticSolverT: TypeAlias = gf.Leapfrog | dfx.SemiImplicitEuler

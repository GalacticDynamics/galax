"""The Leapfrog solver, which now lives in `galax.fields`."""

__all__ = ["Leapfrog", "SymplecticSolverT"]

from typing import TypeAlias

from diffrax import SemiImplicitEuler

from galax.fields import Leapfrog

SymplecticSolverT: TypeAlias = Leapfrog | SemiImplicitEuler

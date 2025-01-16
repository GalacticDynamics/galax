"""Wrappers around `diffrax`."""

__all__ = [
    "DiffEqSolver",
    "VectorizedDenseInterpolation",
]

from .diffeq import DiffEqSolver
from .interp import VectorizedDenseInterpolation

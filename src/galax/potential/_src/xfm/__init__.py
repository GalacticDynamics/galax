"""Transformed Galax Potentials."""

__all__ = [
    "AbstractTransformedPotential",
    "TransformedPotential",
    "TriaxialInThePotential",
]

from .base import AbstractTransformedPotential
from .triaxial import TriaxialInThePotential
from .xop import TransformedPotential

"""Gaussian-density potentials."""

__all__ = [
    "AxisymmetricGaussianPotential",
    "GaussianPotential",
    "TriaxialGaussianPotential",
]

from .axisymmetric import AxisymmetricGaussianPotential
from .base import GaussianPotential
from .triaxial import TriaxialGaussianPotential

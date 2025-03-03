"""Orbits. Private module."""

__all__ = [
    "AbstractOrbit",
    "Orbit",
    # Solve orbits
    "compute_orbit",
    "OrbitSolver",
    # Fields
    "AbstractOrbitField",
    "HamiltonianField",
    "NBodyField",
    # misc
    "plot_components",
    "PhaseSpaceInterpolation",
]

from .api import compute_orbit
from .base import AbstractOrbit
from .field_base import AbstractOrbitField
from .field_hamiltonian import HamiltonianField
from .field_nbody import NBodyField
from .interp import PhaseSpaceInterpolation
from .orbit import Orbit
from .plot_helper import ProxyAbstractOrbit, plot_components
from .solver import OrbitSolver

# Register by import
# isort: split
from . import compute, register_dfx, register_gc  # noqa: F401

ProxyAbstractOrbit.deliver(AbstractOrbit)

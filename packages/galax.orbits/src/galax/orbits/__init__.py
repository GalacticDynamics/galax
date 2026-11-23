""":mod:`galax.orbits`."""

__all__ = [
    "AbstractOrbit",
    "AbstractOrbitField",
    "HamiltonianField",
    "Orbit",
    "OrbitSolver",
    "PhaseSpaceInterpolation",
    "circulation",
    "compute_orbit",
]

from ._src.api import circulation, compute_orbit
from ._src.base import AbstractOrbit
from ._src.field_base import AbstractOrbitField
from ._src.field_hamiltonian import HamiltonianField
from ._src.interp import PhaseSpaceInterpolation
from ._src.orbit import Orbit
from ._src.solver import OrbitSolver

# Register by import
# isort: split
from ._src import circulation as _circulation, compute, register_dfx  # noqa: F401
from ._src.plot_helper import ProxyAbstractOrbit

# Register the experimental `integrate_orbit` dispatches
# isort: split
from ._src import experimental  # noqa: F401

ProxyAbstractOrbit.deliver(AbstractOrbit)

# Cleanup
del ProxyAbstractOrbit

""":mod:`galax.orbits`."""

__all__ = [
    "AbstractOrbit",
    "AbstractOrbitField",
    "HamiltonianField",
    "Orbit",
    "OrbitSolver",
    "PhaseSpaceInterpolation",
    "angular_speed",
    "circulation",
    "compute_orbit",
    "specific_angular_momentum",
]

import warnings

from typing import Any

from ._src.api import circulation, compute_orbit
from ._src.base import AbstractOrbit
from ._src.field_base import AbstractOrbitField
from ._src.field_hamiltonian import HamiltonianField
from ._src.interp import PhaseSpaceInterpolation
from ._src.kinematics import angular_speed, specific_angular_momentum
from ._src.orbit import Orbit
from ._src.solver import OrbitSolver

# Register by import
# isort: split
from . import integrate  # noqa: F401
from ._src import (  # noqa: F401
    circulation as _circulation,
    compute,
    register_dfx,
    register_kinematics,
)
from ._src.plot_helper import ProxyAbstractOrbit

# Register the experimental `integrate_orbit` dispatches
# isort: split
from . import experimental  # noqa: F401

ProxyAbstractOrbit.deliver(AbstractOrbit)


def __getattr__(name: str) -> Any:
    """Serve the deprecated ``omega`` alias of `angular_speed`."""
    if name == "omega":
        warnings.warn(
            "`omega` is deprecated; use `angular_speed`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return angular_speed
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# Cleanup
del ProxyAbstractOrbit

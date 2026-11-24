""":mod:`galax.dynamics`."""

__all__ = [
    # Modules
    "cluster",
    "fields",
    "mockstream",
    "nbody",
    "orbit",
    "plot",
    "examples",
    "integrate",  # TODO: deprecate
    # fields
    "AbstractField",
    "AbstractOrbitField",
    "HamiltonianField",
    "NBodyField",
    # solver
    "AbstractSolver",
    "SolveState",
    "integrate_field",
    # orbit
    "compute_orbit",
    "evaluate_orbit",  # TODO: deprecate
    "Orbit",
    "OrbitSolver",
    # mockstream
    "MockStreamArm",
    "MockStream",
    "MockStreamGenerator",
    # mockstream.df
    "AbstractStreamDF",
    "FardalStreamDF",
    "ChenStreamDF",
    # functions
    "specific_angular_momentum",
    "lagrange_points",
    "tidal_radius",
    "angular_speed",
    "parse_time_specification",
    # ========================
    # Diffraxtra compat
    "DiffEqSolver",
]


import warnings

from typing import Any

from diffraxtra import DiffEqSolver

from . import cluster, examples, fields, integrate, mockstream, nbody, plot
from ._src.cluster import lagrange_points, tidal_radius
from .fields import AbstractField, AbstractOrbitField, HamiltonianField, NBodyField
from .integrate import evaluate_orbit
from .mockstream import (
    AbstractStreamDF,
    ChenStreamDF,
    FardalStreamDF,
    MockStream,
    MockStreamArm,
    MockStreamGenerator,
)
from .orbit import AbstractSolver, Orbit, OrbitSolver, compute_orbit
from .setup_package import load_interop_plugins
from galax.fields import SolveState, integrate_field, parse_time_specification
from galax.orbits import angular_speed, specific_angular_momentum

# isort: split
from ._src import experimental  # noqa: F401

# Interoperability with third-party libraries. Importing a registered module is
# what performs its `plum` dispatch registration; entry points let separately
# installed distributions extend `galax.dynamics` without it knowing they exist.
load_interop_plugins("galax.dynamics.interop")

del load_interop_plugins


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

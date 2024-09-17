""":mod:`galax.dynamics`."""

__all__ = [
    # Modules
    "integrate",
    "mockstream",
    # orbit, et al.
    "Orbit",
    # integrate
    "evaluate_orbit",
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
]


from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics", RUNTIME_TYPECHECKER):
    from . import integrate, mockstream
    from ._src.funcs import (
        lagrange_points,
        specific_angular_momentum,
        tidal_radius,
    )
    from ._src.integrate.funcs import evaluate_orbit
    from ._src.mockstream import MockStream, MockStreamArm, MockStreamGenerator
    from ._src.mockstream.df import AbstractStreamDF, ChenStreamDF, FardalStreamDF
    from ._src.orbit import Orbit


# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

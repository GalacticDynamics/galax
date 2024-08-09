""":mod:`galax.dynamics`."""

__all__ = [
    # Modules
    "integrate",
    "mockstream",
    # orbit, et al.
    "AbstractOrbit",
    "Orbit",
    "InterpolatedOrbit",
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
]


from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics", RUNTIME_TYPECHECKER):
    from . import integrate, mockstream
    from ._dynamics.base import AbstractOrbit
    from ._dynamics.integrate.funcs import evaluate_orbit
    from ._dynamics.mockstream import MockStream, MockStreamArm, MockStreamGenerator
    from ._dynamics.mockstream.df import AbstractStreamDF, ChenStreamDF, FardalStreamDF
    from ._dynamics.orbit import InterpolatedOrbit, Orbit


# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

""":mod:`galax.dynamics`."""

__all__ = [
    # Modules
    "fields",
    "integrate",
    "mockstream",
    "plot",
    # integrate
    "evaluate_orbit",
    "Orbit",
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
    "omega",
]


from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics", RUNTIME_TYPECHECKER):
    from . import fields, integrate, mockstream, plot
    from ._src.api import omega, specific_angular_momentum
    from ._src.cluster.funcs import lagrange_points, tidal_radius
    from ._src.orbit import Orbit
    from .integrate import evaluate_orbit
    from .mockstream import (
        AbstractStreamDF,
        ChenStreamDF,
        FardalStreamDF,
        MockStream,
        MockStreamArm,
        MockStreamGenerator,
    )

    #
    # isort: split
    from ._src import register_api


# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER, register_api

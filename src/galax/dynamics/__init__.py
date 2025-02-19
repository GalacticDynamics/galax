""":mod:`galax.dynamics`."""

__all__ = [
    # Modules
    "fields",
    "solve",
    "integrate",  # TODO: deprecate
    "mockstream",
    "plot",
    "cluster",
    # solve
    "compute_orbit",
    "evaluate_orbit",  # TODO: deprecate
    "Orbit",
    "AbstractSolver",
    "DynamicsSolver",
    # mockstream
    "StreamSimulator",
    "MockStreamArm",
    "MockStream",
    "MockStreamGenerator",  # TODO: deprecate
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
    from . import cluster, fields, integrate, mockstream, plot
    from ._src.api import omega, specific_angular_momentum
    from ._src.cluster import lagrange_points, tidal_radius
    from ._src.orbit import Orbit, compute_orbit
    from .integrate import evaluate_orbit
    from .mockstream import (
        AbstractStreamDF,
        ChenStreamDF,
        FardalStreamDF,
        MockStream,
        MockStreamArm,
        MockStreamGenerator,
        StreamSimulator,
    )
    from .solve import AbstractSolver, DynamicsSolver

    #
    # isort: split
    from ._src import register_api
    from ._src.dynamics import register_gc


# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER, register_api, register_gc

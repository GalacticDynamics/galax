""":mod:`galax.dynamics`."""

__all__ = [
    # Modules
    "cluster",
    "fields",
    "mockstream",
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
    "parse_time_specification",
    # ========================
    # Diffraxtra compat
    "DiffEqSolver",
]


from galax.setup_package import install_import_hook

with install_import_hook("galax.dynamics"):
    from diffraxtra import DiffEqSolver

    from . import cluster, examples, fields, integrate, mockstream, plot
    from ._src.api import omega, specific_angular_momentum
    from ._src.cluster import lagrange_points, tidal_radius
    from ._src.parsetime import parse_time_specification
    from ._src.solver import AbstractSolver, SolveState, integrate_field
    from .fields import AbstractField, AbstractOrbitField, HamiltonianField, NBodyField
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
    from .orbit import AbstractSolver, Orbit, OrbitSolver, compute_orbit

    # isort: split
    from ._src import experimental  # noqa: F401

    # isort: split
    from ._src import register_api


# Cleanup
del install_import_hook, register_api

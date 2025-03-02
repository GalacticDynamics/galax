""":mod:`galax.dynamics`."""

__all__ = [
    # Modules
    "fields",
    "orbit",
    "integrate",  # TODO: deprecate
    "mockstream",
    "plot",
    "cluster",
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
    "omega",
    "parse_time_specification",
    # ========================
    # Diffraxtra compat
    "DiffEqSolver",
]


from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics", RUNTIME_TYPECHECKER):
    from diffraxtra import DiffEqSolver

    from . import cluster, fields, integrate, mockstream, plot
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
    )
    from .orbit import AbstractSolver, Orbit, OrbitSolver, compute_orbit

    # isort: split
    from ._src import register_api
    from ._src.orbit import register_gc


# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER, register_api, register_gc

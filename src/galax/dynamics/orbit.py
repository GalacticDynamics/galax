""":mod:`galax.dynamics.solve`."""

__all__ = [
    "compute_orbit",
    "parse_time_specification",
    "AbstractSolver",
    "OrbitSolver",
    "Orbit",
    "PhaseSpaceInterpolation",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.solve", RUNTIME_TYPECHECKER):
    from ._src.orbit import Orbit, OrbitSolver, PhaseSpaceInterpolation, compute_orbit
    from ._src.parsetime import parse_time_specification
    from ._src.solver import AbstractSolver

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

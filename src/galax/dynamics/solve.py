""":mod:`galax.dynamics.solve`."""

__all__ = [
    "compute_orbit",
    "parse_time_specification",
    "AbstractSolver",
    "SolveState",
    "OrbitSolver",
    "DiffEqSolver",
    "VectorizedDenseInterpolation",
    "Orbit",
    "PhaseSpaceInterpolation",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.solve", RUNTIME_TYPECHECKER):
    from diffraxtra import DiffEqSolver, VectorizedDenseInterpolation

    from ._src.orbit import Orbit, OrbitSolver, PhaseSpaceInterpolation, compute_orbit
    from ._src.parsetime import parse_time_specification
    from ._src.solver import AbstractSolver, SolveState

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

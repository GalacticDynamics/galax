""":mod:`galax.dynamics.solve`."""

__all__ = [
    "evaluate_orbit",
    "parse_time_specification",
    "AbstractSolver",
    "SolveState",
    "DynamicsSolver",
    "DiffEqSolver",
    "VectorizedDenseInterpolation",
    "Orbit",
    "PhaseSpaceInterpolation",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.solve", RUNTIME_TYPECHECKER):
    from diffraxtra import DiffEqSolver, VectorizedDenseInterpolation

    from ._src.dynamics import DynamicsSolver, parse_time_specification
    from ._src.integrate import evaluate_orbit
    from ._src.orbit import Orbit, PhaseSpaceInterpolation
    from ._src.solver import AbstractSolver, SolveState

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

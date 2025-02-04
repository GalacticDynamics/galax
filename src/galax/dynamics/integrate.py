""":mod:`galax.dynamics.integrate`."""

__all__ = [
    "evaluate_orbit",
    "Integrator",
    "Interpolant",
    "parse_time_specification",
    "DiffEqSolver",
    "VectorizedDenseInterpolation",
    "AbstractSolver",
    "DynamicsSolver",
    "InterpolatedPhaseSpacePosition",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.integrate", RUNTIME_TYPECHECKER):
    from ._src.diffeq import DiffEqSolver, VectorizedDenseInterpolation
    from ._src.dynamics import DynamicsSolver, parse_time_specification
    from ._src.integrate import (
        Integrator,
        Interpolant,
        InterpolatedPhaseSpacePosition,
        evaluate_orbit,
    )
    from ._src.solver import AbstractSolver

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

""":mod:`galax.dynamics.integrate`."""

__all__ = [
    "evaluate_orbit",
    "Integrator",
    "Interpolant",
    "parse_time_specification",
    "DiffEqSolver",
    "AbstractSolver",
    "DynamicsSolver",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.integrate", RUNTIME_TYPECHECKER):
    from ._src.integrate import Integrator, Interpolant, evaluate_orbit
    from ._src.solve import (
        AbstractSolver,
        DiffEqSolver,
        DynamicsSolver,
        parse_time_specification,
    )

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

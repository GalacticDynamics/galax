""":mod:`galax.fields`."""

__all__ = [
    "AbstractField",
    "AbstractSolver",
    "SolveState",
    "integrate_field",
    "parse_time_specification",
]

from .setup_package import install_import_hook

with install_import_hook("galax.fields"):
    from ._src.fields import AbstractField
    from ._src.parsetime import parse_time_specification
    from ._src.solver import AbstractSolver, SolveState, integrate_field

# Cleanup
del install_import_hook

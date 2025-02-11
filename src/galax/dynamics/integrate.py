""":mod:`galax.dynamics.integrate`."""

__all__ = [
    "evaluate_orbit",
    "Integrator",
    "PhaseSpaceInterpolation",
    "parse_time_specification",
    "InterpolatedPhaseSpaceCoordinate",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.integrate", RUNTIME_TYPECHECKER):
    from ._src.dynamics import parse_time_specification
    from ._src.legacy import (
        Integrator,
        InterpolatedPhaseSpaceCoordinate,
        evaluate_orbit,
    )
    from ._src.orbit import PhaseSpaceInterpolation

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

""":mod:`galax.dynamics.integrate`."""

__all__ = [
    "evaluate_orbit",
    "Integrator",
    "PhaseSpaceInterpolation",
    "parse_time_specification",
    "InterpolatedPhaseSpaceCoordinate",
]

from galax.setup_package import install_import_hook

with install_import_hook("galax.dynamics.integrate"):
    from ._src.legacy import (
        Integrator,
        InterpolatedPhaseSpaceCoordinate,
        evaluate_orbit,
    )
    from ._src.orbit import PhaseSpaceInterpolation
    from ._src.parsetime import parse_time_specification

# Cleanup
del install_import_hook

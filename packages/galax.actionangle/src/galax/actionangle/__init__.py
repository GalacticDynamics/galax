""":mod:`galax.actionangle`."""

__all__ = [
    "ActionsAnglesFrequencies",
    "harmonic_oscillator_actions_angles_freqs",
    "isochrone_actions_angles_freqs",
    "staeckel_fudge_delta",
]

from .setup_package import install_import_hook

with install_import_hook("galax.actionangle"):
    from ._src.api import (
        harmonic_oscillator_actions_angles_freqs,
        isochrone_actions_angles_freqs,
        staeckel_fudge_delta,
    )
    from ._src.result import ActionsAnglesFrequencies

# Cleanup
del install_import_hook

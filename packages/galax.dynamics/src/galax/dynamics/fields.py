""":mod:`galax.dynamics.fields`."""

__all__ = [
    "AbstractField",
    "AbstractOrbitField",
    "HamiltonianField",
    "NBodyField",
]

from .setup_package import install_import_hook

with install_import_hook("galax.dynamics.fields"):
    from galax.fields import AbstractField
    from galax.nbody import NBodyField
    from galax.orbits import AbstractOrbitField, HamiltonianField

# Cleanup
del install_import_hook

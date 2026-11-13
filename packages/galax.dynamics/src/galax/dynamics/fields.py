""":mod:`galax.dynamics.fields`."""

__all__ = [
    "AbstractField",
    "AbstractOrbitField",
    "HamiltonianField",
    "NBodyField",
]

from .setup_package import install_import_hook

with install_import_hook("galax.dynamics.fields"):
    from galax.fields import (
        AbstractField,
        AbstractOrbitField,
        HamiltonianField,
        NBodyField,
    )

# Cleanup
del install_import_hook

""":mod:`galax.nbody`."""

__all__ = ["NBodyField"]

from .setup_package import install_import_hook

with install_import_hook("galax.nbody"):
    from ._src.field import NBodyField

# Cleanup
del install_import_hook

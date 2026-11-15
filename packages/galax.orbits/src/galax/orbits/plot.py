""":mod:`galax.orbits.plot`."""

__all__ = ["plot_components"]

from .setup_package import install_import_hook

with install_import_hook("galax.orbits.plot"):
    from ._src.plot_helper import plot_components

# Cleanup
del install_import_hook

"""``galax.potential.plot`` module."""

__all__ = [
    "plot_components",
]

from galax.setup_package import install_import_hook

with install_import_hook("galax.dynamics.plot"):
    from ._src.orbit import plot_components

# Cleanup
del install_import_hook

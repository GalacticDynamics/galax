"""``galax.potential.plot`` module."""

__all__ = [
    "plot_components",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.plot", RUNTIME_TYPECHECKER):
    from ._src.orbit import plot_components

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

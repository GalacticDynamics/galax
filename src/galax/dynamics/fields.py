""":mod:`galax.dynamics.fields`."""

__all__ = ["AbstractDynamicsField", "HamiltonianField", "NBodyField"]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.fields", RUNTIME_TYPECHECKER):
    from ._src.fields import AbstractDynamicsField, HamiltonianField, NBodyField

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

""":mod:`galax.dynamics.fields`."""

__all__ = [
    "AbstractField",
    "AbstractOrbitField",
    "HamiltonianField",
    "NBodyField",
    # System-specific fields
    "MWandLMCField",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.fields", RUNTIME_TYPECHECKER):
    from ._src.builtin import MWandLMCField
    from ._src.fields import AbstractField
    from ._src.orbit.field_base import AbstractOrbitField
    from ._src.orbit.field_hamiltonian import HamiltonianField
    from ._src.orbit.field_nbody import NBodyField

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

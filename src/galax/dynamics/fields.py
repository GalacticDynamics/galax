""":mod:`galax.dynamics.fields`."""

__all__ = [
    "AbstractField",
    "AbstractOrbitField",
    "HamiltonianField",
    "NBodyField",
    # System-specific fields
    "RigidMWandLMCField",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.fields", RUNTIME_TYPECHECKER):
    from ._src.example import RigidMWandLMCField
    from ._src.fields import AbstractField
    from ._src.orbit.field_base import AbstractOrbitField
    from ._src.orbit.field_hamiltonian import HamiltonianField
    from ._src.orbit.field_nbody import NBodyField

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

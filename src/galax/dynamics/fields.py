""":mod:`galax.dynamics.fields`."""

__all__ = [
    "AbstractField",
    "AbstractOrbitField",
    "HamiltonianField",
    "NBodyField",
]

from galax.setup_package import install_import_hook

with install_import_hook("galax.dynamics.fields"):
    from ._src.fields import AbstractField
    from ._src.orbit.field_base import AbstractOrbitField
    from ._src.orbit.field_hamiltonian import HamiltonianField
    from ._src.orbit.field_nbody import NBodyField

# Cleanup
del install_import_hook

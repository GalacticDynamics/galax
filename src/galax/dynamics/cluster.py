""":mod:`galax.dynamics.cluster`."""

__all__ = [
    # Modules
    "radius",
    # Solvers
    "MassSolver",
    # Fields
    "MassVectorField",
    "AbstractMassField",
    "UserMassField",
    "ConstantMassField",
    # Events
    "MassBelowThreshold",
    # Functions
    "lagrange_points",
    "tidal_radius",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.fields", RUNTIME_TYPECHECKER):
    from ._src.cluster import (
        AbstractMassField,
        ConstantMassField,
        MassBelowThreshold,
        MassSolver,
        MassVectorField,
        UserMassField,
        lagrange_points,
        radius,
        tidal_radius,
    )

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

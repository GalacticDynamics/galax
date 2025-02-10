""":mod:`galax.dynamics.cluster`."""

__all__ = [
    # Modules
    "radius",
    "relax_time",
    # Solvers
    "MassSolver",
    # Fields
    "MassVectorField",
    "AbstractMassField",
    "CustomMassField",
    "ConstantMass",
    "Baumgardt1998MassLoss",
    # Events
    "MassBelowThreshold",
    # Functions
    "lagrange_points",
    "tidal_radius",
    "relaxation_time",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.fields", RUNTIME_TYPECHECKER):
    from ._src.cluster import (
        AbstractMassField,
        Baumgardt1998MassLoss,
        ConstantMass,
        CustomMassField,
        MassBelowThreshold,
        MassSolver,
        MassVectorField,
        lagrange_points,
        radius,
        relax_time,
        relaxation_time,
        tidal_radius,
    )

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

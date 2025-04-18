""":mod:`galax.dynamics.cluster`."""

__all__ = [
    # Modules
    "radius",
    "relax_time",
    # Solvers
    "MassSolver",
    # Fields
    "MassVectorField",
    "AbstractMassRateField",
    "CustomMassRateField",
    "ZeroMassRate",
    "ConstantMassRate",
    "Baumgardt1998MassLossRate",
    # Events
    "MassBelowThreshold",
    # Sample
    "ReleaseTimeSampler",
    # Functions
    "lagrange_points",
    "tidal_radius",
    "relaxation_time",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.fields", RUNTIME_TYPECHECKER):
    from ._src.cluster import (
        AbstractMassRateField,
        Baumgardt1998MassLossRate,
        ConstantMassRate,
        CustomMassRateField,
        MassBelowThreshold,
        MassSolver,
        MassVectorField,
        ReleaseTimeSampler,
        ZeroMassRate,
        lagrange_points,
        radius,
        relax_time,
        relaxation_time,
        tidal_radius,
    )

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

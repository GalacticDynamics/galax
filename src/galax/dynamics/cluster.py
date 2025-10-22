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

from galax.setup_package import install_import_hook

with install_import_hook("galax.dynamics.fields"):
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
del install_import_hook

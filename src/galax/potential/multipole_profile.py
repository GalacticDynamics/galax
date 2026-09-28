"""`galax.potential.multipole_profile`.

The harmonic primitives these are built from -- projection, the radial
Poisson solve, the splines -- are in `galax.potential.harmonic`.
"""

__all__ = [
    "AbstractMultipoleProfilePotential",
    "MultipoleProfilePotential",
    "build_expansion",
    "expansion_density",
    "expansion_potential",
]

from ._src.builtin.multipole_profile import (
    AbstractMultipoleProfilePotential,
    MultipoleProfilePotential,
    build_expansion,
    expansion_density,
    expansion_potential,
)

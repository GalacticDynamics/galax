"""`galax.dynamics.examples`.

Galax packages some example dynamics systems. These are useful, but do not
capture some (or many) real aspects of the systems they model. For example the
`galax.dynamics.examples.RigidMWandLMCField` is a simple model of the Milky Way
and the Large Magellanic Cloud, treated as rigid potentials acting on each
other's centroids. For this reason we package these as examples.

"""

__all__ = [
    "RigidMWandLMCField",
    "make_mw_lmc_potential",
    "radial_velocity_dispersion_helper",
]

from galax.setup_package import install_import_hook

with install_import_hook("galax.dynamics.examples"):
    from ._src.examples.mw_lmc import (
        RigidMWandLMCField,
        make_mw_lmc_potential,
        radial_velocity_dispersion_helper,
    )

# Cleanup
del install_import_hook

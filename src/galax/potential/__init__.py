""":mod:`galax.potential`."""

__all__ = [
    # Modules
    "io",
    "params",
    "plot",
    # ABCs
    "AbstractPotential",
    "AbstractSinglePotential",
    "AbstractCompositePotential",
    # composite
    "CompositePotential",
    # Pre-composited
    "AbstractPreCompositedPotential",
    # xfm
    "AbstractTransformedPotential",
    "FlattenedInThePotential",
    "TransformedPotential",
    "TriaxialInThePotential",
    "TranslatedPotential",
    # funcs
    "potential",
    "gradient",
    "laplacian",
    "density",
    "hessian",
    "acceleration",
    "tidal_tensor",
    "local_circular_velocity",
    "spherical_mass_enclosed",
    "dpotential_dr",
    "d2potential_dr2",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.potential", RUNTIME_TYPECHECKER):
    from . import io, params, plot
    from ._src.api import (
        acceleration,
        d2potential_dr2,
        density,
        dpotential_dr,
        gradient,
        hessian,
        laplacian,
        local_circular_velocity,
        potential,
        spherical_mass_enclosed,
        tidal_tensor,
    )
    from ._src.base import AbstractPotential
    from ._src.base_multi import (
        AbstractCompositePotential,
        AbstractPreCompositedPotential,
    )
    from ._src.base_single import AbstractSinglePotential
    from ._src.builtin import *
    from ._src.builtin import __all__ as _builtin_all
    from ._src.composite import CompositePotential
    from ._src.xfm import (
        AbstractTransformedPotential,
        FlattenedInThePotential,
        TransformedPotential,
        TranslatedPotential,
        TriaxialInThePotential,
    )

    # Register functions by module import
    # isort: split
    from ._src import register_funcs

__all__ = __all__ + list(_builtin_all)

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER, register_funcs

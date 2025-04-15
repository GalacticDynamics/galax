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
    # builtin
    "LongMuraliBarPotential",
    "MonariEtAl2016BarPotential",
    "KuzminPotential",
    "MiyamotoNagaiPotential",
    "MN3ExponentialPotential",
    "MN3Sech2Potential",
    "HarmonicOscillatorPotential",
    "HenonHeilesPotential",
    "SatohPotential",
    "LMJ09LogarithmicPotential",
    "LogarithmicPotential",
    "AbstractMultipolePotential",
    "MultipoleInnerPotential",
    "MultipoleOuterPotential",
    "MultipolePotential",
    "LeeSutoTriaxialNFWPotential",
    "NFWPotential",
    "TriaxialNFWPotential",
    "Vogelsberger08TriaxialNFWPotential",
    "NullPotential",
    "BurkertPotential",
    "HernquistPotential",
    "IsochronePotential",
    "JaffePotential",
    "KeplerPotential",
    "PlummerPotential",
    "PowerLawCutoffPotential",
    "StoneOstriker15Potential",
    "TriaxialHernquistPotential",
    "HardCutoffNFWPotential",
    # Pre-composited
    "AbstractPreCompositedPotential",
    "BovyMWPotential2014",
    "LM10Potential",
    "MilkyWayPotential",
    "MilkyWayPotential2022",
    # xfm
    "AbstractTransformedPotential",
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
    from ._src.builtin import (
        AbstractMultipolePotential,
        BovyMWPotential2014,
        BurkertPotential,
        HardCutoffNFWPotential,
        HarmonicOscillatorPotential,
        HenonHeilesPotential,
        HernquistPotential,
        IsochronePotential,
        JaffePotential,
        KeplerPotential,
        KuzminPotential,
        LeeSutoTriaxialNFWPotential,
        LM10Potential,
        LMJ09LogarithmicPotential,
        LogarithmicPotential,
        LongMuraliBarPotential,
        MilkyWayPotential,
        MilkyWayPotential2022,
        MiyamotoNagaiPotential,
        MN3ExponentialPotential,
        MN3Sech2Potential,
        MonariEtAl2016BarPotential,
        MultipoleInnerPotential,
        MultipoleOuterPotential,
        MultipolePotential,
        NFWPotential,
        NullPotential,
        PlummerPotential,
        PowerLawCutoffPotential,
        SatohPotential,
        StoneOstriker15Potential,
        TriaxialHernquistPotential,
        TriaxialNFWPotential,
        Vogelsberger08TriaxialNFWPotential,
    )
    from ._src.composite import CompositePotential
    from ._src.xfm import (
        AbstractTransformedPotential,
        TransformedPotential,
        TranslatedPotential,
        TriaxialInThePotential,
    )

    # Register functions by module import
    # isort: split
    from ._src import register_funcs


# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER, register_funcs

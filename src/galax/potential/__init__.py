""":mod:`galax.potential`."""

__all__ = [
    # Modules
    "io",
    "params",
    "plot",
    # base
    "AbstractPotential",
    # core
    "AbstractSinglePotential",
    # composite
    "AbstractCompositePotential",
    "CompositePotential",
    "BarPotential",
    "LongMuraliBarPotential",
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
    "BovyMWPotential2014",
    "LM10Potential",
    "MilkyWayPotential",
    "MilkyWayPotential2022",
    "BurkertPotential",
    "HernquistPotential",
    "IsochronePotential",
    "JaffePotential",
    "KeplerPotential",
    "PlummerPotential",
    "PowerLawCutoffPotential",
    "StoneOstriker15Potential",
    "TriaxialHernquistPotential",
    # frame
    "PotentialFrame",
    # funcs
    "potential",
    "gradient",
    "laplacian",
    "density",
    "hessian",
    "acceleration",
    "tidal_tensor",
    "circular_velocity",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.potential", RUNTIME_TYPECHECKER):
    from . import io, params, plot
    from ._src.base import AbstractPotential
    from ._src.base_multi import AbstractCompositePotential
    from ._src.base_single import AbstractSinglePotential
    from ._src.builtin import (
        AbstractMultipolePotential,
        BarPotential,
        BovyMWPotential2014,
        BurkertPotential,
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
    from ._src.frame import PotentialFrame
    from ._src.funcs import (
        acceleration,
        circular_velocity,
        density,
        gradient,
        hessian,
        laplacian,
        potential,
        tidal_tensor,
    )


# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

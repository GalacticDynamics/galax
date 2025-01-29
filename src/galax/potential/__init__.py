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
    # builtin
    "BurkertPotential",
    "HarmonicOscillatorPotential",
    "HenonHeilesPotential",
    "HernquistPotential",
    "IsochronePotential",
    "JaffePotential",
    "KeplerPotential",
    "KuzminPotential",
    "MiyamotoNagaiPotential",
    "MN3ExponentialPotential",
    "MN3Sech2Potential",
    "NullPotential",
    "PlummerPotential",
    "PowerLawCutoffPotential",
    "SatohPotential",
    "StoneOstriker15Potential",
    "TriaxialHernquistPotential",
    # bars
    "BarPotential",
    "LongMuraliBarPotential",
    # logarithmic
    "LogarithmicPotential",
    "LMJ09LogarithmicPotential",
    # multipole
    "AbstractMultipolePotential",
    "MultipoleInnerPotential",
    "MultipoleOuterPotential",
    "MultipolePotential",
    # nfw
    "NFWPotential",
    "LeeSutoTriaxialNFWPotential",
    "TriaxialNFWPotential",
    "Vogelsberger08TriaxialNFWPotential",
    # special
    "BovyMWPotential2014",
    "LM10Potential",
    "MilkyWayPotential",
    "MilkyWayPotential2022",
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
    from ._src.api import (
        acceleration,
        circular_velocity,
        density,
        gradient,
        hessian,
        laplacian,
        potential,
        tidal_tensor,
    )
    from ._src.base import AbstractPotential
    from ._src.base_multi import AbstractCompositePotential
    from ._src.base_single import AbstractSinglePotential
    from ._src.builtin.bars import BarPotential, LongMuraliBarPotential
    from ._src.builtin.builtin import (
        BurkertPotential,
        HarmonicOscillatorPotential,
        HenonHeilesPotential,
        HernquistPotential,
        IsochronePotential,
        JaffePotential,
        KeplerPotential,
        KuzminPotential,
        MiyamotoNagaiPotential,
        MN3ExponentialPotential,
        MN3Sech2Potential,
        NullPotential,
        PlummerPotential,
        PowerLawCutoffPotential,
        SatohPotential,
        StoneOstriker15Potential,
        TriaxialHernquistPotential,
    )
    from ._src.builtin.logarithmic import (
        LMJ09LogarithmicPotential,
        LogarithmicPotential,
    )
    from ._src.builtin.multipole import (
        AbstractMultipolePotential,
        MultipoleInnerPotential,
        MultipoleOuterPotential,
        MultipolePotential,
    )
    from ._src.builtin.nfw import (
        LeeSutoTriaxialNFWPotential,
        NFWPotential,
        TriaxialNFWPotential,
        Vogelsberger08TriaxialNFWPotential,
    )
    from ._src.builtin.special import (
        BovyMWPotential2014,
        LM10Potential,
        MilkyWayPotential,
        MilkyWayPotential2022,
    )
    from ._src.composite import CompositePotential
    from ._src.frame import PotentialFrame

    # Register functions by module import
    # isort: split
    from ._src import register_funcs


# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER, register_funcs

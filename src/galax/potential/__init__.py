""":mod:`galax.potential`."""

__all__ = [
    # Modules
    "io",
    "params",
    "plot",
    # base
    "AbstractPotentialBase",
    # core
    "AbstractPotential",
    # composite
    "AbstractCompositePotential",
    "CompositePotential",
    # builtin
    "BurkertPotential",
    "HernquistPotential",
    "IsochronePotential",
    "JaffePotential",
    "KeplerPotential",
    "KuzminPotential",
    "MiyamotoNagaiPotential",
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
    from ._potential.base import AbstractPotentialBase
    from ._potential.builtin.bars import BarPotential, LongMuraliBarPotential
    from ._potential.builtin.builtin import (
        BurkertPotential,
        HernquistPotential,
        IsochronePotential,
        JaffePotential,
        KeplerPotential,
        KuzminPotential,
        MiyamotoNagaiPotential,
        NullPotential,
        PlummerPotential,
        PowerLawCutoffPotential,
        SatohPotential,
        StoneOstriker15Potential,
        TriaxialHernquistPotential,
    )
    from ._potential.builtin.logarithmic import (
        LMJ09LogarithmicPotential,
        LogarithmicPotential,
    )
    from ._potential.builtin.multipole import (
        AbstractMultipolePotential,
        MultipoleInnerPotential,
        MultipoleOuterPotential,
        MultipolePotential,
    )
    from ._potential.builtin.nfw import (
        LeeSutoTriaxialNFWPotential,
        NFWPotential,
        TriaxialNFWPotential,
        Vogelsberger08TriaxialNFWPotential,
    )
    from ._potential.builtin.special import (
        BovyMWPotential2014,
        LM10Potential,
        MilkyWayPotential,
    )
    from ._potential.composite import AbstractCompositePotential, CompositePotential
    from ._potential.core import AbstractPotential
    from ._potential.frame import PotentialFrame
    from ._potential.funcs import (
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

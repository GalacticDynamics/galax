"""Interoperability."""

__all__ = ["gala_to_galax"]

from functools import singledispatch

import numpy as np
from gala.potential import (
    CompositePotential as GalaCompositePotential,
    HernquistPotential as GalaHernquistPotential,
    IsochronePotential as GalaIsochronePotential,
    KeplerPotential as GalaKeplerPotential,
    MilkyWayPotential as GalaMilkyWayPotential,
    MiyamotoNagaiPotential as GalaMiyamotoNagaiPotential,
    NFWPotential as GalaNFWPotential,
    NullPotential as GalaNullPotential,
    PotentialBase as GalaPotentialBase,
)

from galax.potential._potential.base import AbstractPotentialBase
from galax.potential._potential.builtin import (
    HernquistPotential,
    IsochronePotential,
    KeplerPotential,
    MiyamotoNagaiPotential,
    NFWPotential,
)
from galax.potential._potential.composite import CompositePotential
from galax.potential._potential.core import AbstractPotential
from galax.potential._potential.special import MilkyWayPotential

##############################################################################
# GALA -> GALAX


def _static_at_origin(pot: GalaPotentialBase, /) -> bool:
    return pot.R is None and np.array_equal(pot.origin, (0, 0, 0))


@singledispatch
def gala_to_galax(pot: GalaPotentialBase, /) -> AbstractPotentialBase:
    """Convert a :mod:`gala` potential to a :mod:`galax` potential.

    Parameters
    ----------
    pot :  :class:`~gala.potential.PotentialBase`
        :mod:`gala` potential.

    Returns
    -------
    gala_pot : :class:`~galax.potential.AbstractPotentialBase`
        :mod:`galax` potential.
    """
    msg = (
        "`gala_to_galax` does not have a registered function to convert "
        f"{pot.__class__.__name__!r} to a `galax.AbstractPotentialBase` instance."
    )
    raise NotImplementedError(msg)


# -----------------------------------------------------------------------------
# General rules


@gala_to_galax.register
def _gala_to_galax_composite(pot: GalaCompositePotential, /) -> CompositePotential:
    """Convert a Gala CompositePotential to a Galax potential."""
    return CompositePotential(**{k: gala_to_galax(p) for k, p in pot.items()})


_GALA_TO_GALAX_REGISTRY: dict[type[GalaPotentialBase], type[AbstractPotential]] = {
    GalaHernquistPotential: HernquistPotential,
    GalaIsochronePotential: IsochronePotential,
    GalaKeplerPotential: KeplerPotential,
    GalaMiyamotoNagaiPotential: MiyamotoNagaiPotential,
}


@gala_to_galax.register(GalaHernquistPotential)
@gala_to_galax.register(GalaIsochronePotential)
@gala_to_galax.register(GalaKeplerPotential)
@gala_to_galax.register(GalaMiyamotoNagaiPotential)
@gala_to_galax.register(GalaNullPotential)
def _gala_to_galax_hernquist(pot: GalaPotentialBase, /) -> AbstractPotential:
    """Convert a Gala HernquistPotential to a Galax potential."""
    if not _static_at_origin(pot):
        msg = "Galax does not support rotating or offset potentials."
        raise TypeError(msg)
    return _GALA_TO_GALAX_REGISTRY[type(pot)](**pot.parameters, units=pot.units)


# -----------------------------------------------------------------------------
# Builtin potentials


@gala_to_galax.register
def _gala_to_galax_nfw(pot: GalaNFWPotential, /) -> NFWPotential:
    """Convert a Gala NFWPotential to a Galax potential."""
    if not _static_at_origin(pot):
        msg = "Galax does not support rotating or offset potentials."
        raise TypeError(msg)
    params = pot.parameters
    return NFWPotential(
        m=params["m"], r_s=params["r_s"], softening_length=0, units=pot.units
    )


# -----------------------------------------------------------------------------
# MW potentials


@gala_to_galax.register
def _gala_to_galax_mwpotential(pot: GalaMilkyWayPotential, /) -> MilkyWayPotential:
    """Convert a Gala MilkyWayPotential to a Galax potential."""
    if not all(_static_at_origin(p) for p in pot.values()):
        msg = "Galax does not support rotating or offset potentials."
        raise TypeError(msg)

    return MilkyWayPotential(
        disk={k: pot["disk"].parameters[k] for k in ("m", "a", "b")},
        halo={k: pot["halo"].parameters[k] for k in ("m", "r_s")},
        bulge={k: pot["bulge"].parameters[k] for k in ("m", "c")},
        nucleus={k: pot["nucleus"].parameters[k] for k in ("m", "c")},
    )

"""Interoperability."""

__all__ = ["gala_to_galax"]

from functools import singledispatch

import numpy as np
from gala.potential import (
    CompositePotential as GalaCompositePotential,
    HernquistPotential as GalaHernquistPotential,
    IsochronePotential as GalaIsochronePotential,
    KeplerPotential as GalaKeplerPotential,
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
    NullPotential,
)
from galax.potential._potential.composite import CompositePotential

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


@gala_to_galax.register
def _gala_to_galax_composite(pot: GalaCompositePotential, /) -> CompositePotential:
    """Convert a Gala CompositePotential to a Galax potential."""
    return CompositePotential(**{k: gala_to_galax(p) for k, p in pot.items()})


@gala_to_galax.register
def _gala_to_galax_hernquist(pot: GalaHernquistPotential, /) -> HernquistPotential:
    """Convert a Gala HernquistPotential to a Galax potential."""
    if not _static_at_origin(pot):
        msg = "Galax does not support rotating or offset potentials."
        raise TypeError(msg)
    params = pot.parameters
    return HernquistPotential(m=params["m"], c=params["c"], units=pot.units)


@gala_to_galax.register
def _gala_to_galax_isochrone(pot: GalaIsochronePotential, /) -> IsochronePotential:
    """Convert a Gala IsochronePotential to a Galax potential."""
    if not _static_at_origin(pot):
        msg = "Galax does not support rotating or offset potentials."
        raise TypeError(msg)
    params = pot.parameters
    return IsochronePotential(m=params["m"], b=params["b"], units=pot.units)


@gala_to_galax.register
def _gala_to_galax_kepler(pot: GalaKeplerPotential, /) -> KeplerPotential:
    """Convert a Gala KeplerPotential to a Galax potential."""
    if not _static_at_origin(pot):
        msg = "Galax does not support rotating or offset potentials."
        raise TypeError(msg)
    params = pot.parameters
    return KeplerPotential(m=params["m"], units=pot.units)


@gala_to_galax.register
def _gala_to_galax_miyamotonagi(
    pot: GalaMiyamotoNagaiPotential, /
) -> MiyamotoNagaiPotential:
    """Convert a Gala MiyamotoNagaiPotential to a Galax potential."""
    if not _static_at_origin(pot):
        msg = "Galax does not support rotating or offset potentials."
        raise TypeError(msg)
    params = pot.parameters
    return MiyamotoNagaiPotential(
        m=params["m"], a=params["a"], b=params["b"], units=pot.units
    )


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


@gala_to_galax.register
def _gala_to_galax_nullpotential(pot: GalaNullPotential, /) -> NullPotential:
    """Convert a Gala NullPotential to a Galax potential."""
    if not _static_at_origin(pot):
        msg = "Galax does not support rotating or offset potentials."
        raise TypeError(msg)
    return NullPotential(units=pot.units)

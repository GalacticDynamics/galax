"""Interoperability."""

__all__ = ["gala_to_galax"]

from functools import singledispatch
from typing import TypeVar

try:  # TODO: less hacky way of supporting optional dependencies
    import pytest
except ImportError:
    pass
else:
    _ = pytest.importorskip("gala")

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
from gala.units import DimensionlessUnitSystem as GalaDimensionlessUnitSystem

import coordinax.operators as cxo
from coordinax.operators import IdentityOperator
from unxt import Quantity

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
from galax.potential._potential.core import AbstractPotential
from galax.potential._potential.frame import PotentialFrame
from galax.potential._potential.special import MilkyWayPotential

##############################################################################
# GALA -> GALAX


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

    Examples
    --------
    The required imports for the examples below are:

    >>> import astropy.units as u
    >>> import gala.potential as gp
    >>> import gala.units as gu
    >>> import galax.potential as gpx

    Going in alphabetical order...

    Composite Potential:

    >>> gpot = gp.CompositePotential(
    ...     disk=gp.MiyamotoNagaiPotential(m=1e11, a=6.5, b=0.26, units=gu.galactic),
    ...     halo=gp.NFWPotential(m=1e12, r_s=20, units=gu.galactic),
    ... )
    >>> gpx.io.gala_to_galax(gpot)
    CompositePotential({'disk': MiyamotoNagaiPotential( ... ),
                        'halo': NFWPotential( ... )})

    Hernquist potential:

    >>> gpot = gp.HernquistPotential(m=1e11 * u.Msun, c=10 * u.kpc, units=gu.galactic)
    >>> gpx.io.gala_to_galax(gpot)
    HernquistPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      m=ConstantParameter(unit=Unit("solMass"), value=f64[]),
      c=ConstantParameter(unit=Unit("kpc"), value=f64[])
    )

    Isochrone potential:

    >>> gpot = gp.IsochronePotential(m=1e11 * u.Msun, b=10 * u.kpc, units=gu.galactic)
    >>> gpx.io.gala_to_galax(gpot)
    IsochronePotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      m=ConstantParameter(unit=Unit("solMass"), value=f64[]),
      b=ConstantParameter(unit=Unit("kpc"), value=f64[])
    )

    Kepler potential:

    >>> gpot = gp.KeplerPotential(m=1e11 * u.Msun, units=gu.galactic)
    >>> gpx.io.gala_to_galax(gpot)
    KeplerPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      m=ConstantParameter(unit=Unit("solMass"), value=f64[])
    )

    Milky Way potential:

    >>> gpot = gp.MilkyWayPotential()
    >>> gpx.io.gala_to_galax(gpot)
    MilkyWayPotential({'disk': MiyamotoNagaiPotential( ... ),
                       'halo': NFWPotential( ... ),
                       'bulge': HernquistPotential( ... ),
                       'nucleus': HernquistPotential( ... )})

    MiyamotoNagai potential:

    >>> gpot = gp.MiyamotoNagaiPotential(m=1e11, a=6.5, b=0.26, units=gu.galactic)
    >>> gpx.io.gala_to_galax(gpot)
    MiyamotoNagaiPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      m=ConstantParameter(unit=Unit("solMass"), value=f64[]),
      a=ConstantParameter(unit=Unit("kpc"), value=f64[]),
      b=ConstantParameter(unit=Unit("kpc"), value=f64[])
    )

    NFW potential:

    >>> gpot = gp.NFWPotential(m=1e12, r_s=20, units=gu.galactic)
    >>> gpx.io.gala_to_galax(gpot)
    NFWPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      m=ConstantParameter(unit=Unit("solMass"), value=f64[]),
      r_s=ConstantParameter(unit=Unit("kpc"), value=f64[]),
      softening_length=0
    )

    Null potential:

    >>> gpot = gp.NullPotential()
    >>> gpx.io.gala_to_galax(gpot)
    NullPotential(units=DimensionlessUnitSystem())
    """
    msg = (
        "`gala_to_galax` does not have a registered function to convert "
        f"{pot.__class__.__name__!r} to a `galax.AbstractPotentialBase` instance."
    )
    raise NotImplementedError(msg)


# -----------------------
# Helper functions

PT = TypeVar("PT", bound=AbstractPotentialBase)


def _get_frame(pot: GalaPotentialBase, /) -> cxo.AbstractOperator:
    frame = cxo.GalileanSpatialTranslationOperator(
        Quantity(pot.origin, unit=pot.units["length"])
    )
    if pot.R is not None:
        frame = cxo.GalileanRotationOperator(pot.R) | frame
    return cxo.simplify_op(frame)


def _apply_frame(frame: cxo.AbstractOperator, pot: PT, /) -> PT | PotentialFrame:
    return pot if isinstance(frame, IdentityOperator) else PotentialFrame(pot, frame)


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
def _gala_to_galax_registered(
    gala: GalaPotentialBase, /
) -> AbstractPotential | PotentialFrame:
    """Convert a Gala HernquistPotential to a Galax potential."""
    if isinstance(gala.units, GalaDimensionlessUnitSystem):
        msg = "Galax does not support converting dimensionless units."
        raise TypeError(msg)

    pot = _GALA_TO_GALAX_REGISTRY[type(gala)](**gala.parameters, units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


# -----------------------------------------------------------------------------
# Builtin potentials


@gala_to_galax.register
def _gala_to_galax_null(_: GalaNullPotential, /) -> NullPotential:
    """Convert a Gala NullPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import galax.potential as gpx

    >>> gpot = gp.NullPotential()
    >>> gpx.io.gala_to_galax(gpot)
    NullPotential(units=DimensionlessUnitSystem())

    """
    return NullPotential(units=DimensionlessUnitSystem())


@gala_to_galax.register
def _gala_to_galax_nfw(gala: GalaNFWPotential, /) -> NFWPotential | PotentialFrame:
    """Convert a Gala NFWPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import galax.potential as gpx

    >>> gpot = gp.NFWPotential(m=1e12, r_s=20, units=gu.galactic)
    >>> gpx.io.gala_to_galax(gpot)
    NFWPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      m=ConstantParameter(unit=Unit("solMass"), value=f64[]),
      r_s=ConstantParameter(unit=Unit("kpc"), value=f64[]),
      softening_length=0
    )

    """
    params = gala.parameters
    pot = NFWPotential(
        m=params["m"], r_s=params["r_s"], softening_length=0, units=gala.units
    )
    return _apply_frame(_get_frame(gala), pot)


# -----------------------------------------------------------------------------
# MW potentials


@gala_to_galax.register
def _gala_to_galax_mw(pot: GalaMilkyWayPotential, /) -> MilkyWayPotential:
    """Convert a Gala MilkyWayPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import galax.potential as gpx

    >>> gpot = gp.MilkyWayPotential()
    >>> gpx.io.gala_to_galax(gpot)
    MilkyWayPotential({'disk': MiyamotoNagaiPotential( ... ),
                       'halo': NFWPotential( ... ),
                       'bulge': HernquistPotential( ... ),
                       'nucleus': HernquistPotential( ... )})

    """
    return MilkyWayPotential(
        disk=gala_to_galax(pot["disk"]),
        halo=gala_to_galax(pot["halo"]),
        bulge=gala_to_galax(pot["bulge"]),
        nucleus=gala_to_galax(pot["nucleus"]),
    )

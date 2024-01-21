"""Interoperability."""

__all__ = ["galax_to_gala"]

from functools import singledispatch

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
from gala.units import UnitSystem as GalaUnitSystem, dimensionless as gala_dimensionless

from galax.potential._potential.base import AbstractPotentialBase
from galax.potential._potential.builtin import (
    BarPotential,
    HernquistPotential,
    IsochronePotential,
    KeplerPotential,
    MiyamotoNagaiPotential,
    NFWPotential,
    NullPotential,
)
from galax.potential._potential.composite import CompositePotential
from galax.potential._potential.param import ConstantParameter
from galax.units import DimensionlessUnitSystem, UnitSystem

##############################################################################
# UnitSystem


def galax_to_gala_units(units: UnitSystem, /) -> GalaUnitSystem:
    if isinstance(units, DimensionlessUnitSystem):
        return gala_dimensionless
    return GalaUnitSystem(units)


##############################################################################
# GALAX -> GALA


# TODO: this can be removed when AbstractPotential gets a `parameters`
# attribute that is a dict whose keys are the names of the parameters.
def _all_constant_parameters(
    pot: "AbstractPotentialBase",
    *params: str,
) -> bool:
    return all(isinstance(getattr(pot, name), ConstantParameter) for name in params)


# TODO: add an argument to specify how to handle time-dependent parameters.
#       Gala potentials are not time-dependent, so we need to specify how to
#       handle time-dependent Galax parameters.
@singledispatch
def galax_to_gala(pot: AbstractPotentialBase, /) -> GalaPotentialBase:
    """Convert a Galax potential to a Gala potential.

    Parameters
    ----------
    pot : :class:`~galax.potential.AbstractPotentialBase`
        Galax potential.

    Returns
    -------
    gala_pot : :class:`~gala.potential.PotentialBase`
        Gala potential.
    """
    msg = (
        "`galax_to_gala` does not have a registered function to convert "
        f"{pot.__class__.__name__!r} to a `gala.PotentialBase` instance."
    )
    raise NotImplementedError(msg)


@galax_to_gala.register
def _galax_to_gala_composite(pot: CompositePotential, /) -> GalaCompositePotential:
    """Convert a Galax CompositePotential to a Gala potential."""
    return GalaCompositePotential(**{k: galax_to_gala(p) for k, p in pot.items()})


@galax_to_gala.register
def _galax_to_gala_bar(pot: BarPotential, /) -> GalaPotentialBase:
    """Convert a Galax BarPotential to a Gala potential."""
    raise NotImplementedError  # TODO: implement


@galax_to_gala.register
def _galax_to_gala_hernquist(pot: HernquistPotential, /) -> GalaHernquistPotential:
    """Convert a Galax HernquistPotential to a Gala potential."""
    if not _all_constant_parameters(pot, "m", "c"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return GalaHernquistPotential(
        m=pot.m(0) * pot.units["mass"],
        c=pot.c(0) * pot.units["length"],
        units=galax_to_gala_units(pot.units),
    )


@galax_to_gala.register
def _galax_to_gala_isochrone(pot: IsochronePotential, /) -> GalaIsochronePotential:
    """Convert a Galax IsochronePotential to a Gala potential."""
    if not _all_constant_parameters(pot, "m", "b"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return GalaIsochronePotential(
        m=pot.m(0) * pot.units["mass"],
        b=pot.b(0) * pot.units["length"],  # TODO: fix the mismatch
        units=galax_to_gala_units(pot.units),
    )


@galax_to_gala.register
def _galax_to_gala_kepler(pot: KeplerPotential, /) -> GalaKeplerPotential:
    """Convert a Galax KeplerPotential to a Gala potential."""
    if not _all_constant_parameters(pot, "m"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return GalaKeplerPotential(
        m=pot.m(0) * pot.units["mass"], units=galax_to_gala_units(pot.units)
    )


@galax_to_gala.register
def _galax_to_gala_miyamotonagi(
    pot: MiyamotoNagaiPotential, /
) -> GalaMiyamotoNagaiPotential:
    """Convert a Galax MiyamotoNagaiPotential to a Gala potential."""
    if not _all_constant_parameters(pot, "m", "a", "b"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return GalaMiyamotoNagaiPotential(
        m=pot.m(0) * pot.units["mass"],
        a=pot.a(0) * pot.units["length"],
        b=pot.b(0) * pot.units["length"],
        units=galax_to_gala_units(pot.units),
    )


@galax_to_gala.register
def _galax_to_gala_nfw(pot: NFWPotential, /) -> GalaNFWPotential:
    """Convert a Galax NFWPotential to a Gala potential."""
    if not _all_constant_parameters(pot, "m", "r_s"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    if pot.softening_length != 0:
        msg = "Gala does not support softening."
        raise TypeError(msg)

    return GalaNFWPotential(
        m=pot.m(0) * pot.units["mass"],
        r_s=pot.r_s(0) * pot.units["length"],
        units=galax_to_gala_units(pot.units),
    )


@galax_to_gala.register
def _galax_to_gala_nullpotential(pot: NullPotential, /) -> GalaNullPotential:
    """Convert a Galax NullPotential to a Gala potential."""
    return GalaNullPotential(units=galax_to_gala_units(pot.units))

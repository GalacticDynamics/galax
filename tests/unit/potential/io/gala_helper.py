"""Interoperability."""

__all__ = ["galax_to_gala"]

from functools import singledispatch

from astropy.units import Quantity as APYQuantity
from gala.potential import (
    CompositePotential as GalaCompositePotential,
    LeeSutoTriaxialNFWPotential as GalaLeeSutoTriaxialNFWPotential,
    MilkyWayPotential as GalaMilkyWayPotential,
    NFWPotential as GalaNFWPotential,
    PotentialBase as GalaPotentialBase,
)
from gala.units import UnitSystem as GalaUnitSystem, dimensionless as gala_dimensionless
from plum import convert

import quaxed.array_api as xp
from unxt import Quantity
from unxt.unitsystems import DimensionlessUnitSystem, UnitSystem

import galax.potential as gp
from galax.potential._potential.io.gala import _GALA_TO_GALAX_REGISTRY

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
    pot: "gp.AbstractPotentialBase",
    *params: str,
) -> bool:
    return all(isinstance(getattr(pot, name), gp.ConstantParameter) for name in params)


# TODO: add an argument to specify how to handle time-dependent parameters.
#       Gala potentials are not time-dependent, so we need to specify how to
#       handle time-dependent Galax parameters.
@singledispatch
def galax_to_gala(pot: gp.AbstractPotentialBase, /) -> GalaPotentialBase:
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


# -----------------------------------------------------------------------------
# General rules


@galax_to_gala.register
def _galax_to_gala_composite(pot: gp.CompositePotential, /) -> GalaCompositePotential:
    """Convert a Galax CompositePotential to a Gala potential."""
    return GalaCompositePotential(**{k: galax_to_gala(p) for k, p in pot.items()})


_GALAX_TO_GALA_REGISTRY: dict[type[gp.AbstractPotential], type[GalaPotentialBase]] = {
    v: k for k, v in _GALA_TO_GALAX_REGISTRY.items()
}


@galax_to_gala.register(gp.HernquistPotential)
@galax_to_gala.register(gp.IsochronePotential)
@galax_to_gala.register(gp.KeplerPotential)
@galax_to_gala.register(gp.MiyamotoNagaiPotential)
@galax_to_gala.register(gp.NullPotential)
def _galax_to_gala_abstractpotential(pot: gp.AbstractPotential, /) -> GalaPotentialBase:
    """Convert a Galax AbstractPotential to a Gala potential."""
    if not _all_constant_parameters(pot, *pot.parameters.keys()):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return _GALAX_TO_GALA_REGISTRY[type(pot)](
        **{
            k: convert(getattr(pot, k)(0), APYQuantity)
            for (k, f) in type(pot).parameters.items()
        },
        units=galax_to_gala_units(pot.units),
    )


# -----------------------------------------------------------------------------
# Builtin potentials


@galax_to_gala.register
def _galax_to_gala_bar(pot: gp.BarPotential, /) -> GalaPotentialBase:
    """Convert a Galax BarPotential to a Gala potential."""
    raise NotImplementedError  # TODO: implement


@galax_to_gala.register
def _galax_to_gala_nfw(pot: gp.NFWPotential, /) -> GalaNFWPotential:
    """Convert a Galax NFWPotential to a Gala potential."""
    if not _all_constant_parameters(pot, "m", "r_s"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return GalaNFWPotential(
        m=convert(pot.m(0), APYQuantity),
        r_s=convert(pot.r_s(0), APYQuantity),
        units=galax_to_gala_units(pot.units),
    )


@galax_to_gala.register
def _galax_to_gala_leesutotriaxialnfw(
    pot: gp.LeeSutoTriaxialNFWPotential, /
) -> GalaLeeSutoTriaxialNFWPotential:
    """Convert a Galax LeeSutoTriaxialNFWPotential to a Gala potential."""
    if not _all_constant_parameters(pot, "m", "r_s", "a1", "a2", "a3"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    t = Quantity(0.0, pot.units["time"])

    return GalaLeeSutoTriaxialNFWPotential(
        v_c=convert(xp.sqrt(pot.constants["G"] * pot.m(t) / pot.r_s(t)), APYQuantity),
        r_s=convert(pot.r_s(t), APYQuantity),
        a=convert(pot.a1(t), APYQuantity),
        b=convert(pot.a2(t), APYQuantity),
        c=convert(pot.a3(t), APYQuantity),
        units=galax_to_gala_units(pot.units),
    )


@galax_to_gala.register
def _gala_to_galax_mwpotential(pot: gp.MilkyWayPotential, /) -> GalaMilkyWayPotential:
    """Convert a Gala MilkyWayPotential to a Galax potential."""
    return GalaMilkyWayPotential(
        disk={k: getattr(pot["disk"], k)(0) for k in ("m", "a", "b")},
        halo={k: getattr(pot["halo"], k)(0) for k in ("m", "r_s")},
        bulge={k: getattr(pot["bulge"], k)(0) for k in ("m", "c")},
        nucleus={k: getattr(pot["nucleus"], k)(0) for k in ("m", "c")},
    )

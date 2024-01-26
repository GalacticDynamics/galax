"""Interoperability."""

__all__ = ["galax_to_gala"]

from functools import singledispatch

from astropy.units import Quantity
from gala.potential import (
    CompositePotential as GalaCompositePotential,
    MilkyWayPotential as GalaMilkyWayPotential,
    NFWPotential as GalaNFWPotential,
    PotentialBase as GalaPotentialBase,
)
from gala.units import UnitSystem as GalaUnitSystem, dimensionless as gala_dimensionless

from galax.potential import (
    AbstractPotential,
    AbstractPotentialBase,
    BarPotential,
    CompositePotential,
    ConstantParameter,
    HernquistPotential,
    IsochronePotential,
    KeplerPotential,
    MilkyWayPotential,
    MiyamotoNagaiPotential,
    NFWPotential,
    NullPotential,
)
from galax.potential._potential.io.gala import _GALA_TO_GALAX_REGISTRY
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


# -----------------------------------------------------------------------------
# General rules


@galax_to_gala.register
def _galax_to_gala_composite(pot: CompositePotential, /) -> GalaCompositePotential:
    """Convert a Galax CompositePotential to a Gala potential."""
    return GalaCompositePotential(**{k: galax_to_gala(p) for k, p in pot.items()})


_GALAX_TO_GALA_REGISTRY: dict[type[AbstractPotential], type[GalaPotentialBase]] = {
    v: k for k, v in _GALA_TO_GALAX_REGISTRY.items()
}


@galax_to_gala.register(HernquistPotential)
@galax_to_gala.register(IsochronePotential)
@galax_to_gala.register(KeplerPotential)
@galax_to_gala.register(MiyamotoNagaiPotential)
@galax_to_gala.register(NullPotential)
def _galax_to_gala_abstractpotential(pot: AbstractPotential, /) -> GalaPotentialBase:
    """Convert a Galax AbstractPotential to a Gala potential."""
    if not _all_constant_parameters(pot, *pot.parameters.keys()):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return _GALAX_TO_GALA_REGISTRY[type(pot)](
        **{
            k: Quantity(getattr(pot, k)(0), unit=pot.units[str(f.dimensions)])
            for (k, f) in type(pot).parameters.items()
        },
        units=galax_to_gala_units(pot.units),
    )


# -----------------------------------------------------------------------------
# Builtin potentials


@galax_to_gala.register
def _galax_to_gala_bar(pot: BarPotential, /) -> GalaPotentialBase:
    """Convert a Galax BarPotential to a Gala potential."""
    raise NotImplementedError  # TODO: implement


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
def _gala_to_galax_mwpotential(pot: MilkyWayPotential, /) -> GalaMilkyWayPotential:
    """Convert a Gala MilkyWayPotential to a Galax potential."""
    return GalaMilkyWayPotential(
        disk={k: getattr(pot["disk"], k)(0) for k in ("m", "a", "b")},
        halo={k: getattr(pot["halo"], k)(0) for k in ("m", "r_s")},
        bulge={k: getattr(pot["bulge"], k)(0) for k in ("m", "c")},
        nucleus={k: getattr(pot["nucleus"], k)(0) for k in ("m", "c")},
    )

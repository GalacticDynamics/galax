"""Interoperability."""

__all__ = ["galax_to_gala"]

from functools import singledispatch

import gala.potential as gp
from astropy.units import Quantity as APYQuantity
from gala.units import UnitSystem as GalaUnitSystem, dimensionless as gala_dimensionless
from packaging.version import Version
from plum import convert

import quaxed.array_api as xp
from unxt import Quantity
from unxt.unitsystems import AbstractUnitSystem, DimensionlessUnitSystem

import galax.potential as gpx
from galax.potential._potential.io._gala import _GALA_TO_GALAX_REGISTRY
from galax.utils._optional_deps import HAS_GALA

##############################################################################
# UnitSystem


def galax_to_gala_units(units: AbstractUnitSystem, /) -> GalaUnitSystem:
    if isinstance(units, DimensionlessUnitSystem):
        return gala_dimensionless
    return GalaUnitSystem(units)


##############################################################################
# GALAX -> GALA


def _all_constant_parameters(
    pot: "gpx.AbstractPotentialBase",
    *params: str,
) -> bool:
    return all(isinstance(getattr(pot, name), gpx.ConstantParameter) for name in params)


# TODO: add an argument to specify how to handle time-dependent parameters.
#       Gala potentials are not time-dependent, so we need to specify how to
#       handle time-dependent Galax parameters.
@singledispatch
def galax_to_gala(pot: gpx.AbstractPotentialBase, /) -> gp.PotentialBase:
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
        f"{pot.__class__.__name__!r} to a galax potential."
    )
    raise NotImplementedError(msg)


# -----------------------------------------------------------------------------
# General rules


@galax_to_gala.register
def _galax_to_gala_composite(pot: gpx.CompositePotential, /) -> gp.CompositePotential:
    """Convert a Galax CompositePotential to a Gala potential."""
    return gp.CompositePotential(**{k: galax_to_gala(p) for k, p in pot.items()})


_GALAX_TO_GALA_REGISTRY: dict[type[gpx.AbstractPotential], type[gp.PotentialBase]] = {
    v: k for k, v in _GALA_TO_GALAX_REGISTRY.items()
}


@galax_to_gala.register(gpx.IsochronePotential)
@galax_to_gala.register(gpx.KeplerPotential)
@galax_to_gala.register(gpx.KuzminPotential)
@galax_to_gala.register(gpx.MiyamotoNagaiPotential)
@galax_to_gala.register(gpx.PlummerPotential)
@galax_to_gala.register(gpx.PowerLawCutoffPotential)
def _galax_to_gala_abstractpotential(pot: gpx.AbstractPotential, /) -> gp.PotentialBase:
    """Convert a Galax AbstractPotential to a Gala potential."""
    if not _all_constant_parameters(pot, *pot.parameters.keys()):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    # TODO: this is a temporary solution. It would be better to map each
    # potential individually.
    params = {
        k: convert(getattr(pot, k)(0), APYQuantity)
        for (k, f) in type(pot).parameters.items()
    }
    if "m_tot" in params:
        params["m"] = params.pop("m_tot")

    return _GALAX_TO_GALA_REGISTRY[type(pot)](
        **params,
        units=galax_to_gala_units(pot.units),
    )


# -----------------------------------------------------------------------------
# Builtin potentials


@galax_to_gala.register
def _galax_to_gala_bar(pot: gpx.BarPotential, /) -> gp.PotentialBase:
    """Convert a Galax BarPotential to a Gala potential."""
    raise NotImplementedError  # TODO: implement


if HAS_GALA and (Version("1.8.2") <= HAS_GALA):

    @galax_to_gala.register
    def _galax_to_gala_burkert(pot: gpx.BurkertPotential, /) -> gp.BurkertPotential:
        """Convert a Galax BurkertPotential to a Gala potential."""
        if not _all_constant_parameters(pot, "m", "r_s"):
            msg = "Gala does not support time-dependent parameters."
            raise TypeError(msg)

        return gp.BurkertPotential(
            rho=convert(pot.rho0(0), APYQuantity),
            r0=convert(pot.r_s(0), APYQuantity),
            units=galax_to_gala_units(pot.units),
        )


@galax_to_gala.register
def _galax_to_gala_hernquist(pot: gpx.HernquistPotential, /) -> gp.HernquistPotential:
    """Convert a Galax HernquistPotential to a Gala potential."""
    if not _all_constant_parameters(pot, "m_tot", "r_s"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return gp.HernquistPotential(
        m=convert(pot.m_tot(0), APYQuantity),
        c=convert(pot.r_s(0), APYQuantity),
        units=galax_to_gala_units(pot.units),
    )


@galax_to_gala.register
def _galax_to_gala_jaffe(pot: gpx.JaffePotential, /) -> gp.JaffePotential:
    """Convert a Galax JaffePotential to a Gala potential."""
    if not _all_constant_parameters(pot, "m", "r_s"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return gp.JaffePotential(
        m=convert(pot.m(0), APYQuantity),
        c=convert(pot.r_s(0), APYQuantity),
        units=galax_to_gala_units(pot.units),
    )


@galax_to_gala.register
def _galax_to_gala_longmuralibar(
    pot: gpx.LongMuraliBarPotential, /
) -> gp.LongMuraliBarPotential:
    """Convert a Galax LongMuraliBarPotential to a Gala potential."""
    if not _all_constant_parameters(pot, "m_tot", "a", "b", "c", "alpha"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return gp.LongMuraliBarPotential(
        m=convert(pot.m_tot(0), APYQuantity),
        a=convert(pot.a(0), APYQuantity),
        b=convert(pot.b(0), APYQuantity),
        c=convert(pot.c(0), APYQuantity),
        alpha=convert(pot.alpha(0), APYQuantity),
        units=galax_to_gala_units(pot.units),
    )


@galax_to_gala.register
def _galax_to_gala_null(pot: gpx.NullPotential, /) -> gp.NullPotential:
    return gp.NullPotential(
        units=galax_to_gala_units(pot.units),
    )


@galax_to_gala.register
def _galax_to_gala_satoh(pot: gpx.SatohPotential, /) -> gp.SatohPotential:
    """Convert a Galax SatohPotential to a Gala potential."""
    if not _all_constant_parameters(pot, "m_tot", "a", "b"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return gp.SatohPotential(
        m=convert(pot.m_tot(0), APYQuantity),
        a=convert(pot.a(0), APYQuantity),
        b=convert(pot.b(0), APYQuantity),
        units=galax_to_gala_units(pot.units),
    )


@galax_to_gala.register
def _galax_to_gala_stoneostriker15(
    pot: gpx.StoneOstriker15Potential, /
) -> gp.StonePotential:
    """Convert a Galax StoneOstriker15Potential to a Gala potential."""
    if not _all_constant_parameters(pot, "m_tot", "r_c", "r_h"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return gp.StonePotential(
        m=convert(pot.m_tot(0), APYQuantity),
        r_c=convert(pot.r_c(0), APYQuantity),
        r_h=convert(pot.r_h(0), APYQuantity),
        units=galax_to_gala_units(pot.units),
    )


# -----------------------------------------------------------------------------
# Logarithmic potentials


@galax_to_gala.register
def _galax_to_gala_logarithmic(
    pot: gpx.LogarithmicPotential, /
) -> gp.LogarithmicPotential:
    """Convert a Galax LogarithmicPotential to a Gala potential."""
    if not _all_constant_parameters(pot, "v_c", "r_s"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return gp.LogarithmicPotential(
        v_c=convert(pot.v_c(0), APYQuantity),
        r_h=convert(pot.r_s(0), APYQuantity),
        units=galax_to_gala_units(pot.units),
    )


@galax_to_gala.register
def _galax_to_gala_logarithmic(
    pot: gpx.LMJ09LogarithmicPotential, /
) -> gp.LogarithmicPotential:
    """Convert a Galax LogarithmicPotential to a Gala potential."""
    if not _all_constant_parameters(pot, "v_c", "r_s", "q1", "q2", "q3", "phi"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return gp.LogarithmicPotential(
        v_c=convert(pot.v_c(0), APYQuantity),
        r_h=convert(pot.r_s(0), APYQuantity),
        q1=convert(pot.q1(0), APYQuantity),
        q2=convert(pot.q2(0), APYQuantity),
        q3=convert(pot.q3(0), APYQuantity),
        phi=convert(pot.phi(0), APYQuantity),
        units=galax_to_gala_units(pot.units),
    )


# -----------------------------------------------------------------------------
# NFW potentials


@galax_to_gala.register
def _galax_to_gala_nfw(pot: gpx.NFWPotential, /) -> gp.NFWPotential:
    """Convert a Galax NFWPotential to a Gala potential."""
    if not _all_constant_parameters(pot, "m", "r_s"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return gp.NFWPotential(
        m=convert(pot.m(0), APYQuantity),
        r_s=convert(pot.r_s(0), APYQuantity),
        units=galax_to_gala_units(pot.units),
    )


@galax_to_gala.register
def _galax_to_gala_leesutotriaxialnfw(
    pot: gpx.LeeSutoTriaxialNFWPotential, /
) -> gp.LeeSutoTriaxialNFWPotential:
    """Convert a Galax LeeSutoTriaxialNFWPotential to a Gala potential."""
    if not _all_constant_parameters(pot, "m", "r_s", "a1", "a2", "a3"):
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    t = Quantity(0.0, pot.units["time"])

    return gp.LeeSutoTriaxialNFWPotential(
        v_c=convert(xp.sqrt(pot.constants["G"] * pot.m(t) / pot.r_s(t)), APYQuantity),
        r_s=convert(pot.r_s(t), APYQuantity),
        a=convert(pot.a1(t), APYQuantity),
        b=convert(pot.a2(t), APYQuantity),
        c=convert(pot.a3(t), APYQuantity),
        units=galax_to_gala_units(pot.units),
    )


# -----------------------------------------------------------------------------
# Composite potentials


@galax_to_gala.register
def _galax_to_gala_bovymw2014(
    pot: gpx.BovyMWPotential2014, /
) -> gp.BovyMWPotential2014:
    """Convert a Galax BovyMWPotential2014 to a Gala potential."""

    def rename(k: str) -> str:
        match k:
            case "m_tot":
                return "m"
            case _:
                return k

    return gp.BovyMWPotential2014(
        **{
            c: {rename(k): getattr(p, k)(0) for k in p.parameters}
            for c, p in pot.items()
        }
    )


@galax_to_gala.register
def _galax_to_gala_lm10(pot: gpx.LM10Potential, /) -> gp.LM10Potential:
    """Convert a Galax LM10Potential to a Gala potential."""

    def rename(c: str, k: str) -> str:
        match k:
            case "m_tot":
                return "m"
            case "r_s" if c == "halo":
                return "r_h"
            case "r_s" if c == "bulge":
                return "c"
            case _:
                return k

    return gp.LM10Potential(
        **{
            c: {rename(c, k): getattr(p, k)(0) for k in p.parameters}
            for c, p in pot.items()
        }
    )


@galax_to_gala.register
def _galax_to_gala_mwpotential(pot: gpx.MilkyWayPotential, /) -> gp.MilkyWayPotential:
    """Convert a Galax MilkyWayPotential to a Gala potential."""

    def rename(c: str, k: str) -> str:
        match k:
            case "m_tot":
                return "m"
            case "r_s" if c in ("bulge", "nucleus"):
                return "c"
            case _:
                return k

    return gp.MilkyWayPotential(
        **{
            c: {rename(c, k): getattr(p, k)(0) for k in p.parameters}
            for c, p in pot.items()
        }
    )

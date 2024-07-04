"""Interoperability."""

__all__ = ["gala_to_galax", "galax_to_gala"]

from functools import singledispatch
from typing import TypeVar

import gala.potential as gp
from astropy.units import Quantity as APYQuantity
from gala.units import (
    DimensionlessUnitSystem as GalaDimensionlessUnitSystem,
    UnitSystem as GalaUnitSystem,
    dimensionless as gala_dimensionless,
)
from packaging.version import Version
from plum import convert, dispatch

import coordinax.operators as cxo
from coordinax.operators import IdentityOperator
from unxt import Quantity
from unxt.unitsystems import AbstractUnitSystem, DimensionlessUnitSystem

import galax.potential as gpx
from galax.utils._optional_deps import HAS_GALA

##############################################################################
# Hook into general dispatcher


@dispatch
def convert_potential(
    to_: gpx.AbstractPotentialBase | type[gpx.io.GalaxLibrary],  # noqa: ARG001
    from_: gp.CPotentialBase | gp.PotentialBase,
    /,
) -> gpx.AbstractPotentialBase:
    return gala_to_galax(from_)


@dispatch
def convert_potential(
    to_: gp.CPotentialBase | gp.PotentialBase | type[gpx.io.GalaLibrary],  # noqa: ARG001
    from_: gpx.AbstractPotentialBase,
    /,
) -> gp.CPotentialBase | gp.PotentialBase:
    return galax_to_gala(from_)


##############################################################################
# GALAX <-> GALA

# -----------------------
# Helper functions

PT = TypeVar("PT", bound=gpx.AbstractPotentialBase)


def _get_frame(pot: gp.PotentialBase, /) -> cxo.AbstractOperator:
    """Convert a Gala frame to a Galax frame."""
    frame = cxo.GalileanSpatialTranslationOperator(
        Quantity(pot.origin, unit=pot.units["length"])
    )
    if pot.R is not None:
        frame = cxo.GalileanRotationOperator(pot.R) | frame
    return cxo.simplify_op(frame)


def _apply_frame(frame: cxo.AbstractOperator, pot: PT, /) -> PT | gpx.PotentialFrame:
    """Apply a Galax frame to a potential."""
    # A framed Galax potential never simplifies to a frameless potential. This
    # function applies a frame if it is not the identity operator.
    return (
        pot if isinstance(frame, IdentityOperator) else gpx.PotentialFrame(pot, frame)
    )


def _galax_to_gala_units(units: AbstractUnitSystem, /) -> GalaUnitSystem:
    """Convert a Galax unit system to a Gala unit system."""
    # Galax potentials naturally convert Gala unit systems, but Gala potentials
    # do not convert Galax unit systems. This function is used for that purpose.
    if isinstance(units, DimensionlessUnitSystem):
        return gala_dimensionless
    return GalaUnitSystem(units)


def _error_if_not_all_constant_parameters(
    pot: gpx.AbstractPotentialBase, *params: str
) -> None:
    """Check if all parameters are constant."""
    is_time_dep = any(
        not isinstance(getattr(pot, name), gpx.params.ConstantParameter)
        for name in params
    )

    if is_time_dep:
        msg = "Gala does not support time-dependent parameters."
        raise TypeError(msg)

    return


# -----------------------------------------------------------------------------


@singledispatch
def gala_to_galax(pot: gp.PotentialBase, /) -> gpx.AbstractPotentialBase:
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
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    CompositePotential({'disk': MiyamotoNagaiPotential( ... ),
                        'halo': NFWPotential( ... )})

    Hernquist potential:

    >>> gpot = gp.HernquistPotential(m=1e11 * u.Msun, c=10 * u.kpc, units=gu.galactic)
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    HernquistPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ),
      r_s=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) ) )

    Isochrone potential:

    >>> gpot = gp.IsochronePotential(m=1e11 * u.Msun, b=10 * u.kpc, units=gu.galactic)
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    IsochronePotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ),
      b=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) ) )

    Kepler potential:

    >>> gpot = gp.KeplerPotential(m=1e11 * u.Msun, units=gu.galactic)
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    KeplerPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ) )

    >>> gpot = gp.LeeSutoTriaxialNFWPotential(
    ...     v_c=220, r_s=20, a=1, b=0.9, c=0.8, units=gu.galactic )
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    LeeSutoTriaxialNFWPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableMap({'G': ...}),
      m=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ),
      r_s=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) ),
      a1=ConstantParameter( unit=Unit(dimensionless), value=Quantity[...]( value=f64[], unit=Unit(dimensionless) ) ),
      a2=ConstantParameter( unit=Unit(dimensionless), value=Quantity[...]( value=f64[], unit=Unit(dimensionless) ) ),
      a3=ConstantParameter( unit=Unit(dimensionless), value=Quantity[...]( value=f64[], unit=Unit(dimensionless) ) )
    )

    Milky Way potential:

    >>> gpot = gp.MilkyWayPotential()
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    MilkyWayPotential({'disk': MiyamotoNagaiPotential( ... ),
                       'halo': NFWPotential( ... ),
                       'bulge': HernquistPotential( ... ),
                       'nucleus': HernquistPotential( ... )})

    MiyamotoNagai potential:

    >>> gpot = gp.MiyamotoNagaiPotential(m=1e11, a=6.5, b=0.26, units=gu.galactic)
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    MiyamotoNagaiPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ),
      a=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) ),
      b=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) ) )

    NFW potential:

    >>> gpot = gp.NFWPotential(m=1e12, r_s=20, units=gu.galactic)
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    NFWPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableMap({'G': ...}),
      m=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ),
      r_s=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) ) )

    Null potential:

    >>> gpot = gp.NullPotential()
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    NullPotential( units=DimensionlessUnitSystem(),
                   constants=ImmutableMap({'G': ...}) )
    """  # noqa: E501
    msg = (
        "`gala_to_galax` does not have a registered function to convert "
        f"{pot.__class__.__name__!r} to a `galax.AbstractPotentialBase` instance."
    )
    raise NotImplementedError(msg)


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
# Composite potentials


@gala_to_galax.register
def _gala_to_galax_composite(pot: gp.CompositePotential, /) -> gpx.CompositePotential:
    """Convert a Gala CompositePotential to a Galax potential."""
    return gpx.CompositePotential(**{k: gala_to_galax(p) for k, p in pot.items()})


@galax_to_gala.register
def _galax_to_gala_composite(pot: gpx.CompositePotential, /) -> gp.CompositePotential:
    """Convert a Galax CompositePotential to a Gala potential."""
    return gp.CompositePotential(**{k: galax_to_gala(p) for k, p in pot.items()})


# -----------------------------------------------------------------------------
# Builtin potentials


@galax_to_gala.register
def _galax_to_gala_bar(_: gpx.BarPotential, /) -> gp.PotentialBase:
    """Convert a Galax BarPotential to a Gala potential."""
    raise NotImplementedError  # TODO: implement


if HAS_GALA and (Version("1.8.2") <= HAS_GALA):

    @gala_to_galax.register
    def _gala_to_galax_burkert(
        gala: gp.BurkertPotential, /
    ) -> gpx.BurkertPotential | gpx.PotentialFrame:
        """Convert a Gala BurkertPotential to a Galax potential.

        Examples
        --------
        >>> import gala.potential as gp
        >>> import gala.units as gu
        >>> import galax.potential as gpx

        .. invisible-code-block: python

            from packaging.version import Version
            from galax.utils._optional_deps import HAS_GALA
            skip = not HAS_GALA or HAS_GALA < Version("1.8.2")

        .. skip: start if(skip, reason="Requires Gala v1.8.2+")

        >>> gpot = gp.BurkertPotential(rho=4, r0=20, units=gu.galactic)
        >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
        BurkertPotential(
        units=UnitSystem(kpc, Myr, solMass, rad),
        constants=ImmutableMap({'G': ...}),
        m=ConstantParameter( ... ),
        r_s=ConstantParameter( ... )
        )

        .. skip: end
        """
        params = gala.parameters
        pot = gpx.BurkertPotential.from_central_density(
            rho_0=params["rho"], r_s=params["r0"], units=gala.units
        )
        return _apply_frame(_get_frame(gala), pot)

    @galax_to_gala.register
    def _galax_to_gala_burkert(pot: gpx.BurkertPotential, /) -> gp.BurkertPotential:
        """Convert a Galax BurkertPotential to a Gala potential."""
        _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

        return gp.BurkertPotential(
            rho=convert(pot.rho0(0), APYQuantity),
            r0=convert(pot.r_s(0), APYQuantity),
            units=_galax_to_gala_units(pot.units),
        )

# ---------------------------
# Hernquist potentials


@gala_to_galax.register
def _gala_to_galax_hernquist(
    gala: gp.HernquistPotential, /
) -> gpx.HernquistPotential | gpx.PotentialFrame:
    r"""Convert a Gala HernquistPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import gala.units as gu
    >>> import galax.potential as gpx

    >>> gpot = gp.HernquistPotential(m=1e11, c=20, units=gu.galactic)
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    HernquistPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      r_s=ConstantParameter( ... )
    )
    """
    params = gala.parameters
    pot = gpx.HernquistPotential(m_tot=params["m"], r_s=params["c"], units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


@galax_to_gala.register
def _galax_to_gala_hernquist(pot: gpx.HernquistPotential, /) -> gp.HernquistPotential:
    """Convert a Galax HernquistPotential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return gp.HernquistPotential(
        m=convert(pot.m_tot(0), APYQuantity),
        c=convert(pot.r_s(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Isochrone potentials


@gala_to_galax.register
def _gala_to_galax_isochrone(
    gala: gp.IsochronePotential, /
) -> gpx.IsochronePotential | gpx.PotentialFrame:
    """Convert a Gala potential to a Galax potential."""
    if isinstance(gala.units, GalaDimensionlessUnitSystem):
        msg = "Galax does not support converting dimensionless units."
        raise TypeError(msg)

    params = dict(gala.parameters)
    params["m_tot"] = params.pop("m")

    pot = gpx.IsochronePotential(**params, units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


@galax_to_gala.register
def _galax_to_gala_isochrone(pot: gpx.IsochronePotential, /) -> gp.IsochronePotential:
    """Convert a Galax AbstractPotential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    params = {
        k: convert(getattr(pot, k)(0), APYQuantity)
        for (k, f) in type(pot).parameters.items()
    }
    if "m_tot" in params:
        params["m"] = params.pop("m_tot")

    return gp.IsochronePotential(**params, units=_galax_to_gala_units(pot.units))


# ---------------------------
# Jaffe potentials


@gala_to_galax.register
def _gala_to_galax_jaffe(
    gala: gp.JaffePotential, /
) -> gpx.JaffePotential | gpx.PotentialFrame:
    """Convert a Gala JaffePotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import gala.units as gu
    >>> import galax.potential as gpx

    >>> gpot = gp.JaffePotential(m=1e11, c=20, units=gu.galactic)
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    JaffePotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableMap({'G': ...}),
      m=ConstantParameter( ... ),
      r_s=ConstantParameter( ... )
    )
    """
    params = gala.parameters
    pot = gpx.JaffePotential(m=params["m"], r_s=params["c"], units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


@galax_to_gala.register
def _galax_to_gala_jaffe(pot: gpx.JaffePotential, /) -> gp.JaffePotential:
    """Convert a Galax JaffePotential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return gp.JaffePotential(
        m=convert(pot.m(0), APYQuantity),
        c=convert(pot.r_s(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Kepler potentials


@gala_to_galax.register
def _gala_to_galax_kepler(
    gala: gp.KeplerPotential, /
) -> gpx.KeplerPotential | gpx.PotentialFrame:
    """Convert a Gala potential to a Galax potential."""
    if isinstance(gala.units, GalaDimensionlessUnitSystem):
        msg = "Galax does not support converting dimensionless units."
        raise TypeError(msg)

    params = dict(gala.parameters)
    params["m_tot"] = params.pop("m")

    pot = gpx.KeplerPotential(**params, units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


@galax_to_gala.register
def _galax_to_gala_kepler(pot: gpx.KeplerPotential, /) -> gp.KeplerPotential:
    """Convert a Galax AbstractPotential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    params = {
        k: convert(getattr(pot, k)(0), APYQuantity)
        for (k, f) in type(pot).parameters.items()
    }
    if "m_tot" in params:
        params["m"] = params.pop("m_tot")

    return gp.KeplerPotential(**params, units=_galax_to_gala_units(pot.units))


# ---------------------------
# Kuzmin potentials


@gala_to_galax.register
def _gala_to_galax_registered(
    gala: gp.KuzminPotential, /
) -> gpx.KuzminPotential | gpx.PotentialFrame:
    """Convert a Gala potential to a Galax potential."""
    if isinstance(gala.units, GalaDimensionlessUnitSystem):
        msg = "Galax does not support converting dimensionless units."
        raise TypeError(msg)

    params = dict(gala.parameters)
    params["m_tot"] = params.pop("m")

    pot = gpx.KuzminPotential(**params, units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


@galax_to_gala.register
def _galax_to_gala_kuzmin(pot: gpx.KuzminPotential, /) -> gp.KuzminPotential:
    """Convert a Galax AbstractPotential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    params = {
        k: convert(getattr(pot, k)(0), APYQuantity)
        for (k, f) in type(pot).parameters.items()
    }
    if "m_tot" in params:
        params["m"] = params.pop("m_tot")

    return gp.KuzminPotential(**params, units=_galax_to_gala_units(pot.units))


# ---------------------------
# Long & Murali Bar potentials


@gala_to_galax.register
def _gala_to_galax_longmuralibar(
    gala: gp.LongMuraliBarPotential, /
) -> gpx.LongMuraliBarPotential | gpx.PotentialFrame:
    """Convert a Gala LongMuraliBarPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import gala.units as gu
    >>> import galax.potential as gpx

    >>> gpot = gp.LongMuraliBarPotential(m=1e11, a=20, b=10, c=5, units=gu.galactic)
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    LongMuraliBarPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableMap({'G': Quantity...}),
      m_tot=ConstantParameter( ... ),
      a=ConstantParameter( ... ),
      b=ConstantParameter( ... ),
      c=ConstantParameter( ... ),
      alpha=ConstantParameter( ... )
    )
    """
    params = gala.parameters
    pot = gpx.LongMuraliBarPotential(
        m_tot=params["m"],
        a=params["a"],
        b=params["b"],
        c=params["c"],
        alpha=params["alpha"],
        units=gala.units,
    )
    return _apply_frame(_get_frame(gala), pot)


@galax_to_gala.register
def _galax_to_gala_longmuralibar(
    pot: gpx.LongMuraliBarPotential, /
) -> gp.LongMuraliBarPotential:
    """Convert a Galax LongMuraliBarPotential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return gp.LongMuraliBarPotential(
        m=convert(pot.m_tot(0), APYQuantity),
        a=convert(pot.a(0), APYQuantity),
        b=convert(pot.b(0), APYQuantity),
        c=convert(pot.c(0), APYQuantity),
        alpha=convert(pot.alpha(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Miyamoto-Nagai potentials


@gala_to_galax.register
def _gala_to_galax_registered(
    gala: gp.MiyamotoNagaiPotential, /
) -> gpx.MiyamotoNagaiPotential | gpx.PotentialFrame:
    """Convert a Gala potential to a Galax potential."""
    if isinstance(gala.units, GalaDimensionlessUnitSystem):
        msg = "Galax does not support converting dimensionless units."
        raise TypeError(msg)

    params = dict(gala.parameters)
    params["m_tot"] = params.pop("m")

    pot = gpx.MiyamotoNagaiPotential(**params, units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


@galax_to_gala.register
def _galax_to_gala_mn(pot: gpx.MiyamotoNagaiPotential, /) -> gp.MiyamotoNagaiPotential:
    """Convert a Galax AbstractPotential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    params = {
        k: convert(getattr(pot, k)(0), APYQuantity)
        for (k, f) in type(pot).parameters.items()
    }
    if "m_tot" in params:
        params["m"] = params.pop("m_tot")

    return gp.MiyamotoNagaiPotential(**params, units=_galax_to_gala_units(pot.units))


# ---------------------------
# Null potentials


@gala_to_galax.register
def _gala_to_galax_null(pot: gp.NullPotential, /) -> gpx.NullPotential:
    """Convert a Gala NullPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import galax.potential as gpx

    >>> gpot = gp.NullPotential()
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    NullPotential( units=DimensionlessUnitSystem(),
                   constants=ImmutableMap({'G': ...}) )

    """
    return gpx.NullPotential(units=pot.units)


@galax_to_gala.register
def _galax_to_gala_null(pot: gpx.NullPotential, /) -> gp.NullPotential:
    return gp.NullPotential(
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Plummer potentials


@gala_to_galax.register
def _gala_to_galax_registered(
    gala: gp.PlummerPotential, /
) -> gpx.PlummerPotential | gpx.PotentialFrame:
    """Convert a Gala potential to a Galax potential."""
    if isinstance(gala.units, GalaDimensionlessUnitSystem):
        msg = "Galax does not support converting dimensionless units."
        raise TypeError(msg)

    params = dict(gala.parameters)
    params["m_tot"] = params.pop("m")

    pot = gpx.PlummerPotential(**params, units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


@galax_to_gala.register
def _galax_to_gala_plummer(pot: gpx.PlummerPotential, /) -> gp.PlummerPotential:
    """Convert a Galax AbstractPotential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    params = {
        k: convert(getattr(pot, k)(0), APYQuantity)
        for (k, f) in type(pot).parameters.items()
    }
    if "m_tot" in params:
        params["m"] = params.pop("m_tot")

    return gp.PlummerPotential(**params, units=_galax_to_gala_units(pot.units))


# ---------------------------
# PowerLawCutoff potentials


@gala_to_galax.register
def _gala_to_galax_registered(
    gala: gp.PowerLawCutoffPotential, /
) -> gpx.PowerLawCutoffPotential | gpx.PotentialFrame:
    """Convert a Gala potential to a Galax potential."""
    if isinstance(gala.units, GalaDimensionlessUnitSystem):
        msg = "Galax does not support converting dimensionless units."
        raise TypeError(msg)

    params = dict(gala.parameters)
    params["m_tot"] = params.pop("m")

    pot = gpx.PowerLawCutoffPotential(**params, units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


@galax_to_gala.register
def _galax_to_gala_powerlaw(
    pot: gpx.PowerLawCutoffPotential, /
) -> gp.PowerLawCutoffPotential:
    """Convert a Galax AbstractPotential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    params = {
        k: convert(getattr(pot, k)(0), APYQuantity)
        for (k, f) in type(pot).parameters.items()
    }
    if "m_tot" in params:
        params["m"] = params.pop("m_tot")

    return gp.PowerLawCutoffPotential(**params, units=_galax_to_gala_units(pot.units))


# ---------------------------
# Satoh potentials


@gala_to_galax.register
def _gala_to_galax_satoh(
    gala: gp.SatohPotential, /
) -> gpx.SatohPotential | gpx.PotentialFrame:
    """Convert a Gala SatohPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import gala.units as gu
    >>> import galax.potential as gpx

    >>> gpot = gp.SatohPotential(m=1e11, a=20, b=10, units=gu.galactic)
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    SatohPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      a=ConstantParameter( ... ),
      b=ConstantParameter( ... )
    )
    """
    params = gala.parameters
    pot = gpx.SatohPotential(
        m_tot=params["m"], a=params["a"], b=params["b"], units=gala.units
    )
    return _apply_frame(_get_frame(gala), pot)


@galax_to_gala.register
def _galax_to_gala_satoh(pot: gpx.SatohPotential, /) -> gp.SatohPotential:
    """Convert a Galax SatohPotential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return gp.SatohPotential(
        m=convert(pot.m_tot(0), APYQuantity),
        a=convert(pot.a(0), APYQuantity),
        b=convert(pot.b(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Stone & Ostriker potentials


@gala_to_galax.register
def _gala_to_galax_stoneostriker15(
    gala: gp.StonePotential, /
) -> gpx.StoneOstriker15Potential | gpx.PotentialFrame:
    """Convert a Gala StonePotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import gala.units as gu
    >>> import galax.potential as gpx

    >>> gpot = gp.StonePotential(m=1e11, r_c=20, r_h=10, units=gu.galactic)
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    StoneOstriker15Potential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      r_c=ConstantParameter( ... ),
      r_h=ConstantParameter( ... )
    )
    """
    params = gala.parameters
    pot = gpx.StoneOstriker15Potential(
        m_tot=params["m"], r_c=params["r_c"], r_h=params["r_h"], units=gala.units
    )
    return _apply_frame(_get_frame(gala), pot)


@galax_to_gala.register
def _galax_to_gala_stoneostriker15(
    pot: gpx.StoneOstriker15Potential, /
) -> gp.StonePotential:
    """Convert a Galax StoneOstriker15Potential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return gp.StonePotential(
        m=convert(pot.m_tot(0), APYQuantity),
        r_c=convert(pot.r_c(0), APYQuantity),
        r_h=convert(pot.r_h(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# -----------------------------------------------------------------------------
# Logarithmic potentials


@gala_to_galax.register
def _gala_to_galax_logarithmic(
    gala: gp.LogarithmicPotential, /
) -> gpx.LogarithmicPotential | gpx.LMJ09LogarithmicPotential | gpx.PotentialFrame:
    """Convert a Gala LogarithmicPotential to a Galax potential.

    If the flattening or rotation 'phi' is non-zero, the potential is a
    :class:`galax.potential.LMJ09LogarithmicPotential` (or
    :class:`galax.potential.PotentialFrame` wrapper thereof). Otherwise, it is a
    :class:`galax.potential.LogarithmicPotential` (or
    :class:`galax.potential.PotentialFrame` wrapper thereof).

    Examples
    --------
    >>> import gala.potential as gp
    >>> import gala.units as gu
    >>> import galax.potential as gpx

    >>> gpot = gp.LogarithmicPotential(v_c=220, r_h=20, units=gu.galactic)
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    LogarithmicPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableMap({'G': ...}),
      v_c=ConstantParameter( ... ),
      r_s=ConstantParameter( ... )
    )
    """
    params = gala.parameters

    if (
        params["q1"] != 1
        or params["q2"] != 1
        or params["q3"] != 1
        or params["phi"] != 0
    ):
        pot = gpx.LMJ09LogarithmicPotential(
            v_c=params["v_c"],
            r_s=params["r_h"],
            q1=params["q1"],
            q2=params["q2"],
            q3=params["q3"],
            phi=params["phi"],
            units=gala.units,
        )
    else:
        pot = gpx.LogarithmicPotential(
            v_c=params["v_c"], r_s=params["r_h"], units=gala.units
        )

    return _apply_frame(_get_frame(gala), pot)


@galax_to_gala.register
def _galax_to_gala_logarithmic(
    pot: gpx.LogarithmicPotential, /
) -> gp.LogarithmicPotential:
    """Convert a Galax LogarithmicPotential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return gp.LogarithmicPotential(
        v_c=convert(pot.v_c(0), APYQuantity),
        r_h=convert(pot.r_s(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


@galax_to_gala.register
def _galax_to_gala_logarithmic(
    pot: gpx.LMJ09LogarithmicPotential, /
) -> gp.LogarithmicPotential:
    """Convert a Galax LogarithmicPotential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return gp.LogarithmicPotential(
        v_c=convert(pot.v_c(0), APYQuantity),
        r_h=convert(pot.r_s(0), APYQuantity),
        q1=convert(pot.q1(0), APYQuantity),
        q2=convert(pot.q2(0), APYQuantity),
        q3=convert(pot.q3(0), APYQuantity),
        phi=convert(pot.phi(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# -----------------------------------------------------------------------------
# NFW potentials


@gala_to_galax.register
def _gala_to_galax_nfw(
    gala: gp.NFWPotential, /
) -> gpx.NFWPotential | gpx.PotentialFrame:
    """Convert a Gala NFWPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import gala.units as gu
    >>> import galax.potential as gpx

    >>> gpot = gp.NFWPotential(m=1e12, r_s=20, units=gu.galactic)
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    NFWPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableMap({'G': ...}),
      m=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ),
      r_s=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) )
    )

    """  # noqa: E501
    params = gala.parameters
    pot = gpx.NFWPotential(m=params["m"], r_s=params["r_s"], units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


@galax_to_gala.register
def _galax_to_gala_nfw(pot: gpx.NFWPotential, /) -> gp.NFWPotential:
    """Convert a Galax NFWPotential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return gp.NFWPotential(
        m=convert(pot.m(0), APYQuantity),
        r_s=convert(pot.r_s(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


@gala_to_galax.register
def _gala_to_galax_leesutotriaxialnfw(
    pot: gp.LeeSutoTriaxialNFWPotential, /
) -> gpx.LeeSutoTriaxialNFWPotential:
    """Convert a Gala LeeSutoTriaxialNFWPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import gala.units as gu
    >>> import galax.potential as gpx

    >>> gpot = gp.LeeSutoTriaxialNFWPotential(
    ...     v_c=220, r_s=20, a=1, b=0.9, c=0.8, units=gu.galactic )
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    LeeSutoTriaxialNFWPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableMap({'G': ...}),
      m=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ),
      r_s=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) ),
      a1=ConstantParameter( unit=Unit(dimensionless), value=Quantity[...]( value=f64[], unit=Unit(dimensionless) ) ),
      a2=ConstantParameter( unit=Unit(dimensionless), value=Quantity[...]( value=f64[], unit=Unit(dimensionless) ) ),
      a3=ConstantParameter( unit=Unit(dimensionless), value=Quantity[...]( value=f64[], unit=Unit(dimensionless) ) )
    )

    """  # noqa: E501
    units = pot.units
    params = pot.parameters
    G = Quantity(pot.G, units["length"] ** 3 / units["time"] ** 2 / units["mass"])

    return gpx.LeeSutoTriaxialNFWPotential(
        m=params["v_c"] ** 2 * params["r_s"] / G,
        r_s=params["r_s"],
        a1=params["a"],
        a2=params["b"],
        a3=params["c"],
        units=units,
        constants={"G": G},
    )


@galax_to_gala.register
def _galax_to_gala_leesutotriaxialnfw(
    pot: gpx.LeeSutoTriaxialNFWPotential, /
) -> gp.LeeSutoTriaxialNFWPotential:
    """Convert a Galax LeeSutoTriaxialNFWPotential to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    t = Quantity(0.0, pot.units["time"])

    return gp.LeeSutoTriaxialNFWPotential(
        v_c=convert(xp.sqrt(pot.constants["G"] * pot.m(t) / pot.r_s(t)), APYQuantity),
        r_s=convert(pot.r_s(t), APYQuantity),
        a=convert(pot.a1(t), APYQuantity),
        b=convert(pot.a2(t), APYQuantity),
        c=convert(pot.a3(t), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# -----------------------------------------------------------------------------
# MW potentials

# ---------------------------
# Bovy MWPotential2014


@gala_to_galax.register
def _gala_to_galax_bovymw2014(
    pot: gp.BovyMWPotential2014, /
) -> gpx.BovyMWPotential2014:
    """Convert a Gala BovyMWPotential2014 to a Galax potential.

    Examples
    --------
    .. invisible-code-block: python

        from galax.utils._optional_deps import GSL_ENABLED

    .. skip: start if(not GSL_ENABLED, reason="requires GSL")

    >>> import gala.potential as gp
    >>> import galax.potential as gpx

    >>> gpot = gp.BovyMWPotential2014()
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    BovyMWPotential2014({'disk': MiyamotoNagaiPotential( ... ),
                        'bulge': PowerLawCutoffPotential( ... ),
                        'halo': NFWPotential( ... )})

    .. skip: end

    """
    return gpx.BovyMWPotential2014(
        disk=gala_to_galax(pot["disk"]),
        bulge=gala_to_galax(pot["bulge"]),
        halo=gala_to_galax(pot["halo"]),
    )


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


# ---------------------------
# LM10 potentials


@gala_to_galax.register
def _gala_to_galax_lm10(pot: gp.LM10Potential, /) -> gpx.LM10Potential:
    """Convert a Gala LM10Potential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import galax.potential as gpx

    >>> gpot = gp.LM10Potential()
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    LM10Potential({'disk': MiyamotoNagaiPotential( ... ),
                   'bulge': HernquistPotential( ... ),
                   'halo': LMJ09LogarithmicPotential( ... )})

    """
    return gpx.LM10Potential(
        disk=gala_to_galax(pot["disk"]),
        bulge=gala_to_galax(pot["bulge"]),
        halo=gala_to_galax(pot["halo"]),
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


# ---------------------------
# Galax MilkyWayPotential


@gala_to_galax.register
def _gala_to_galax_mw(pot: gp.MilkyWayPotential, /) -> gpx.MilkyWayPotential:
    """Convert a Gala MilkyWayPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import galax.potential as gpx

    >>> gpot = gp.MilkyWayPotential()
    >>> gpx.io.convert_potential(gpx.io.GalaLibrary, gpot)
    MilkyWayPotential({'disk': MiyamotoNagaiPotential( ... ),
                       'halo': NFWPotential( ... ),
                       'bulge': HernquistPotential( ... ),
                       'nucleus': HernquistPotential( ... )})

    """
    return gpx.MilkyWayPotential(
        disk=gala_to_galax(pot["disk"]),
        halo=gala_to_galax(pot["halo"]),
        bulge=gala_to_galax(pot["bulge"]),
        nucleus=gala_to_galax(pot["nucleus"]),
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

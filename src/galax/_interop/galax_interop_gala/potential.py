"""Interoperability with :mod:`gala` potentials."""

__all__ = ["gala_to_galax", "galax_to_gala"]

from typing import TypeVar

import equinox as eqx
import gala.potential as gp
import jax.numpy as jnp
from astropy.units import Quantity as APYQuantity
from gala.units import (
    DimensionlessUnitSystem as GalaDimensionlessUnitSystem,
    UnitSystem as GalaUnitSystem,
    dimensionless as gala_dimensionless,
)
from packaging.version import Version
from plum import convert, dispatch

import coordinax.operators as cxo
import quaxed.array_api as xp
from coordinax.operators import IdentityOperator
from unxt import Quantity
from unxt.unitsystems import AbstractUnitSystem, DimensionlessUnitSystem

import galax.potential as gpx
from galax._interop.optional_deps import OptDeps

##############################################################################
# Hook into general dispatcher


@dispatch
def convert_potential(
    to_: gpx.AbstractPotentialBase | type[gpx.io.GalaxLibrary],  # noqa: ARG001
    from_: gp.CPotentialBase | gp.PotentialBase,
    /,
) -> gpx.AbstractPotentialBase:
    """Convert a :class:`~gala.potential.PotentialBase` to a :class:`~galax.potential.AbstractPotentialBase`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.KeplerPotential(m=1e11, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    KeplerPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( value=Quantity[...](value=f64[], unit=Unit("solMass")) ) )

    """  # noqa: E501
    return gala_to_galax(from_)


@dispatch
def convert_potential(
    to_: gp.CPotentialBase | gp.PotentialBase | type[gpx.io.GalaLibrary],  # noqa: ARG001
    from_: gpx.AbstractPotentialBase,
    /,
) -> gp.CPotentialBase | gp.PotentialBase:
    """Convert a :class:`~galax.potential.AbstractPotentialBase` to a :class:`~gala.potential.PotentialBase`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=Quantity(1e11, "Msun"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <KeplerPotential: m=1.00e+11 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    return galax_to_gala(from_)


# NOTE: this is a bit of type piracy, but since `gala` does not use `plum` and
# so does not support this function, this is totally fine.
@dispatch
def convert_potential(
    to_: gp.CPotentialBase | gp.PotentialBase | type[gpx.io.GalaLibrary],  # noqa: ARG001
    from_: gp.CPotentialBase | gp.PotentialBase,
    /,
) -> gp.CPotentialBase | gp.PotentialBase:
    """Convert a :class:`~galax.potential.AbstractPotentialBase` to itself.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.KeplerPotential(m=1e11, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot) is pot
    True

    """
    return from_


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


# TODO: check if `galax` handles this internally
def _check_gala_units(gala: GalaUnitSystem, /) -> GalaUnitSystem:
    return eqx.error_if(
        gala,
        isinstance(gala, GalaDimensionlessUnitSystem),
        "Galax does not support converting dimensionless units.",
    )


# -----------------------------------------------------------------------------


@dispatch  # type: ignore[misc]
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

    """
    msg = (
        "`gala_to_galax` does not have a registered function to convert "
        f"{pot.__class__.__name__!r} to a `galax.AbstractPotentialBase` instance."
    )
    raise NotImplementedError(msg)


# TODO: add an argument to specify how to handle time-dependent parameters.
#       Gala potentials are not time-dependent, so we need to specify how to
#       handle time-dependent Galax parameters.
@dispatch  # type: ignore[misc]
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


@dispatch  # type: ignore[misc]
def gala_to_galax(pot: gp.CompositePotential, /) -> gpx.CompositePotential:
    """Convert a `gala.potential.CompositePotential` -> `galax.potential.CompositePotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.CompositePotential(
    ...     disk=galap.MiyamotoNagaiPotential(m=1e11, a=6.5, b=0.26, units=galactic),
    ...     halo=galap.NFWPotential(m=1e12, r_s=20, units=galactic),
    ... )
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    CompositePotential({'disk': MiyamotoNagaiPotential( ... ),
                        'halo': NFWPotential( ... )})
    """  # noqa: E501
    return gpx.CompositePotential(**{k: gala_to_galax(p) for k, p in pot.items()})


@dispatch
def galax_to_gala(pot: gpx.CompositePotential, /) -> gp.CompositePotential:
    """Convert a `galax.potential.CompositePotential` -> `gala.potential.CompositePotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.CompositePotential(
    ...     disk=gp.MiyamotoNagaiPotential(m_tot=Quantity(1e11, "Msun"), a=6.5, b=0.26, units="galactic"),
    ...     halo=gp.NFWPotential(m=Quantity(1e12, "Msun"), r_s=20, units="galactic"),
    ... )
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <CompositePotential disk,halo>

    """  # noqa: E501
    return gp.CompositePotential(**{k: galax_to_gala(p) for k, p in pot.items()})


# -----------------------------------------------------------------------------
# Builtin potentials


@dispatch
def galax_to_gala(_: gpx.BarPotential, /) -> gp.PotentialBase:
    """Convert a Galax BarPotential to a Gala potential."""
    raise NotImplementedError  # TODO: implement


if OptDeps.GALA.is_installed and (Version("1.8.2") <= OptDeps.GALA.version):

    @dispatch  # type: ignore[misc]
    def gala_to_galax(
        gala: gp.BurkertPotential, /
    ) -> gpx.BurkertPotential | gpx.PotentialFrame:
        """Convert a `gala.potential.BurkertPotential` to a galax.potential.BurkertPotential.

        Examples
        --------
        >>> import gala.potential as galap
        >>> from gala.units import galactic
        >>> import galax.potential as gp

        .. invisible-code-block: python

            from packaging.version import Version
            from galax._interop.optional_deps import OptDeps
            skip = not OptDeps.GALA.is_installed or OptDeps.GALA.version < Version("1.8.2")

        .. skip: start if(skip, reason="Requires Gala v1.8.2+")

        >>> pot = galap.BurkertPotential(rho=4, r0=20, units=galactic)
        >>> gpx.io.convert_potential(gp.io.GalaxLibrary, pot)
        BurkertPotential(
        units=LTMAUnitSystem( length=Unit("kpc"), ...),
        constants=ImmutableMap({'G': ...}),
        m=ConstantParameter( ... ),
        r_s=ConstantParameter( ... )
        )

        .. skip: end

        """  # noqa: E501
        params = gala.parameters
        pot = gpx.BurkertPotential.from_central_density(
            rho_0=params["rho"], r_s=params["r0"], units=gala.units
        )
        return _apply_frame(_get_frame(gala), pot)

    @dispatch  # type: ignore[misc]
    def galax_to_gala(pot: gpx.BurkertPotential, /) -> gp.BurkertPotential:
        """Convert a `galax.potential.BurkertPotential` to a `gala.potential.BurkertPotential`.

        Examples
        --------
        >>> import gala.potential as galap
        >>> from unxt import Quantity
        >>> import galax.potential as gp

        .. invisible-code-block: python

            from packaging.version import Version
            from galax._interop.optional_deps import OptDeps
            skip = not OptDeps.GALA.is_installed or OptDeps.GALA.version < Version("1.8.2")

        .. skip: start if(skip, reason="Requires Gala v1.8.2+")

        >>> pot = gp.BurkertPotential(m=Quantity(1e11, "Msun"), r_s=Quantity(20, "kpc"), units="galactic")
        >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
        <BurkertPotential: rho=7.82e+06, r0=20.00 (kpc,Myr,solMass,rad)>

        .. skip: end

        """  # noqa: E501
        _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

        return gp.BurkertPotential(
            rho=convert(pot.rho0(0), APYQuantity),
            r0=convert(pot.r_s(0), APYQuantity),
            units=_galax_to_gala_units(pot.units),
        )

# ---------------------------
# Hernquist potentials


@dispatch  # type: ignore[misc]
def gala_to_galax(
    gala: gp.HernquistPotential, /
) -> gpx.HernquistPotential | gpx.PotentialFrame:
    r"""Convert a `gala.potential.HernquistPotential` to a `galax.potential.HernquistPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.HernquistPotential(m=1e11, c=20, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    HernquistPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      r_s=ConstantParameter( ... )
    )
    """  # noqa: E501
    params = gala.parameters
    pot = gpx.HernquistPotential(
        m_tot=params["m"], r_s=params["c"], units=_check_gala_units(gala.units)
    )
    return _apply_frame(_get_frame(gala), pot)


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.HernquistPotential, /) -> gp.HernquistPotential:
    """Convert a `galax.potential.HernquistPotential` to a `gala.potential.HernquistPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.HernquistPotential(m_tot=Quantity(1e11, "Msun"), r_s=Quantity(20, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <HernquistPotential: m=1.00e+11, c=20.00 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return gp.HernquistPotential(
        m=convert(pot.m_tot(0), APYQuantity),
        c=convert(pot.r_s(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Isochrone potentials


@dispatch  # type: ignore[misc]
def gala_to_galax(
    gala: gp.IsochronePotential, /
) -> gpx.IsochronePotential | gpx.PotentialFrame:
    """Convert a `gala.potential.IsochronePotential` to a `galax.potential.IsochronePotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.IsochronePotential(m=1e11, b=10, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    IsochronePotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      b=ConstantParameter( ... )

    """  # noqa: E501
    params = dict(gala.parameters)
    params["m_tot"] = params.pop("m")

    pot = gpx.IsochronePotential(**params, units=_check_gala_units(gala.units))
    return _apply_frame(_get_frame(gala), pot)


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.IsochronePotential, /) -> gp.IsochronePotential:
    """Convert a `galax.potential.IsochronePotential` to a `gala.potential.IsochronePotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.IsochronePotential(m_tot=Quantity(1e11, "Msun"), b=Quantity(10, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <IsochronePotential: m=1.00e+11, b=10.00 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
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


@dispatch  # type: ignore[misc]
def gala_to_galax(
    gala: gp.JaffePotential, /
) -> gpx.JaffePotential | gpx.PotentialFrame:
    """Convert a Gala JaffePotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.JaffePotential(m=1e11, c=20, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    JaffePotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m=ConstantParameter( ... ),
      r_s=ConstantParameter( ... )
    )
    """
    params = gala.parameters
    pot = gpx.JaffePotential(
        m=params["m"], r_s=params["c"], units=_check_gala_units(gala.units)
    )
    return _apply_frame(_get_frame(gala), pot)


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.JaffePotential, /) -> gp.JaffePotential:
    """Convert a `galax.potential.JaffePotential` to a `gala.potential.JaffePotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.JaffePotential(m=Quantity(1e11, "Msun"), r_s=Quantity(20, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <JaffePotential: m=1.00e+11, c=20.00 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return gp.JaffePotential(
        m=convert(pot.m(0), APYQuantity),
        c=convert(pot.r_s(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Kepler potentials


@dispatch  # type: ignore[misc]
def gala_to_galax(
    gala: gp.KeplerPotential, /
) -> gpx.KeplerPotential | gpx.PotentialFrame:
    """Convert a `gala.potential.KeplerPotential` to a `galax.potential.KeplerPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.KeplerPotential(m=1e11, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    KeplerPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( value=Quantity[...](value=f64[], unit=Unit("solMass")) ) )
    """  # noqa: E501
    params = dict(gala.parameters)
    params["m_tot"] = params.pop("m")

    pot = gpx.KeplerPotential(**params, units=_check_gala_units(gala.units))
    return _apply_frame(_get_frame(gala), pot)


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.KeplerPotential, /) -> gp.KeplerPotential:
    """Convert a `galax.potential.KeplerPotential` to a `gala.potential.KeplerPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=Quantity(1e11, "Msun"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <KeplerPotential: m=1.00e+11 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
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


@dispatch  # type: ignore[misc]
def gala_to_galax(
    gala: gp.KuzminPotential, /
) -> gpx.KuzminPotential | gpx.PotentialFrame:
    """Convert a `gala.potential.KuzminPotential` to a `galax.potential.KuzminPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.KuzminPotential(m=1e11, a=20, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    KuzminPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=...,
      a=...
    )

    """  # noqa: E501
    params = dict(gala.parameters)
    params["m_tot"] = params.pop("m")

    pot = gpx.KuzminPotential(**params, units=_check_gala_units(gala.units))
    return _apply_frame(_get_frame(gala), pot)


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.KuzminPotential, /) -> gp.KuzminPotential:
    """Convert a `galax.potential.KuzminPotential` to a `gala.potential.KuzminPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.KuzminPotential(m_tot=Quantity(1e11, "Msun"), a=Quantity(20, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <KuzminPotential: m=1.00e+11, a=20.00 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
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


@dispatch  # type: ignore[misc]
def gala_to_galax(
    gala: gp.LongMuraliBarPotential, /
) -> gpx.LongMuraliBarPotential | gpx.PotentialFrame:
    """Convert a Gala LongMuraliBarPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.LongMuraliBarPotential(m=1e11, a=20, b=10, c=5, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    LongMuraliBarPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
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


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.LongMuraliBarPotential, /) -> gp.LongMuraliBarPotential:
    """Convert a `galax.potential.LongMuraliBarPotential` to a `gala.potential.LongMuraliBarPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.LongMuraliBarPotential(
    ...     m_tot=Quantity(1e11, "Msun"),
    ...     a=Quantity(20, "kpc"),
    ...     b=Quantity(10, "kpc"),
    ...     c=Quantity(5, "kpc"),
    ...     alpha=Quantity(0.1, "rad"),
    ...     units="galactic",
    ... )
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <LongMuraliBarPotential: m=1.00e+11, a=20.00, b=10.00, c=5.00, alpha=0.10 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
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


@dispatch  # type: ignore[misc]
def gala_to_galax(
    gala: gp.MiyamotoNagaiPotential, /
) -> gpx.MiyamotoNagaiPotential | gpx.PotentialFrame:
    """Convert a `gala.potential.MiyamotoNagaiPotential` to a `galax.potential.MiyamotoNagaiPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.MiyamotoNagaiPotential(m=1e11, a=6.5, b=0.26, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    MiyamotoNagaiPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      a=ConstantParameter( ... ),
      b=ConstantParameter( ... )
    )

    """  # noqa: E501
    params = dict(gala.parameters)
    params["m_tot"] = params.pop("m")

    pot = gpx.MiyamotoNagaiPotential(**params, units=_check_gala_units(gala.units))
    return _apply_frame(_get_frame(gala), pot)


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.MiyamotoNagaiPotential, /) -> gp.MiyamotoNagaiPotential:
    """Convert a `galax.potential.MiyamotoNagaiPotential` to a `gala.potential.MiyamotoNagaiPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.MiyamotoNagaiPotential(m_tot=Quantity(1e11, "Msun"), a=Quantity(6.5, "kpc"), b=Quantity(0.26, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <MiyamotoNagaiPotential: m=1.00e+11, a=6.50, b=0.26 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
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


@dispatch  # type: ignore[misc]
def gala_to_galax(pot: gp.NullPotential, /) -> gpx.NullPotential:
    """Convert a `gala.potential.NullPotential` to a `galax.potential.NullPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> import galax.potential as gp

    >>> pot = gp.NullPotential()
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    NullPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...})
    )

    """
    return gpx.NullPotential(units=pot.units)


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.NullPotential, /) -> gp.NullPotential:
    """Convert a `galax.potential.NullPotential` to a `gala.potential.NullPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> import galax.potential as gp

    >>> pot = gp.NullPotential()
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <NullPotential:  (kpc,Myr,solMass,rad)>

    """
    return gp.NullPotential(
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Plummer potentials


@dispatch  # type: ignore[misc]
def gala_to_galax(
    gala: gp.PlummerPotential, /
) -> gpx.PlummerPotential | gpx.PotentialFrame:
    """Convert a `gala.potential.PlummerPotential` to a `galax.potential.PlummerPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.PlummerPotential(m=1e11, b=1, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    PlummerPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      b=ConstantParameter( ... )
    )

    """  # noqa: E501
    params = dict(gala.parameters)
    params["m_tot"] = params.pop("m")

    pot = gpx.PlummerPotential(**params, units=_check_gala_units(gala.units))
    return _apply_frame(_get_frame(gala), pot)


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.PlummerPotential, /) -> gp.PlummerPotential:
    """Convert a `galax.potential.PlummerPotential` to a `gala.potential.PlummerPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.PlummerPotential(m_tot=Quantity(1e11, "Msun"), b=Quantity(10, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <PlummerPotential: m=1.00e+11, b=10.00 (kpc,Myr,solMass,rad)>
    """  # noqa: E501
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


@dispatch  # type: ignore[misc]
def gala_to_galax(
    gala: gp.PowerLawCutoffPotential, /
) -> gpx.PowerLawCutoffPotential | gpx.PotentialFrame:
    """Convert a `gala.potential.PowerLawCutoffPotential` to a `galax.potential.PowerLawCutoffPotential`.

    Examples
    --------
    .. invisible-code-block: python

        from galax._interop.optional_deps import GSL_ENABLED

    .. skip: start if(not GSL_ENABLED, reason="requires GSL")

    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.PowerLawCutoffPotential(m=1e11, alpha=1.8, r_c=20, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    PowerLawCutoffPotential(
        units=LTMAUnitSystem( length=Unit("kpc"), ...),
        constants=ImmutableMap({'G': ...}),
        m_tot=ConstantParameter( ... ),
        alpha=ConstantParameter( ... ),
        r_c=ConstantParameter( ... )
    )

    .. skip: end

    """  # noqa: E501
    params = dict(gala.parameters)
    params["m_tot"] = params.pop("m")

    pot = gpx.PowerLawCutoffPotential(**params, units=_check_gala_units(gala.units))
    return _apply_frame(_get_frame(gala), pot)


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.PowerLawCutoffPotential, /) -> gp.PowerLawCutoffPotential:
    """Convert a `galax.potential.PowerLawCutoffPotential` to a `gala.potential.PowerLawCutoffPotential`.

    Examples
    --------
    .. invisible-code-block: python

        from galax._interop.optional_deps import GSL_ENABLED

    .. skip: start if(not GSL_ENABLED, reason="requires GSL")

    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.PowerLawCutoffPotential(m_tot=Quantity(1e11, "Msun"), alpha=1.8, r_c=Quantity(20, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <PowerLawCutoffPotential: m=1.00e+11, alpha=1.80, r_c=20.00 (kpc,Myr,solMass,rad)>

    .. skip: end
    """  # noqa: E501
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


@dispatch  # type: ignore[misc]
def gala_to_galax(
    gala: gp.SatohPotential, /
) -> gpx.SatohPotential | gpx.PotentialFrame:
    """Convert a Gala SatohPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.SatohPotential(m=1e11, a=20, b=10, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    SatohPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
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


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.SatohPotential, /) -> gp.SatohPotential:
    """Convert a `galax.potential.SatohPotential` to a `gala.potential.SatohPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.SatohPotential(m_tot=Quantity(1e11, "Msun"), a=Quantity(20, "kpc"), b=Quantity(10, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <SatohPotential: m=1.00e+11, a=20.00, b=10.00 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return gp.SatohPotential(
        m=convert(pot.m_tot(0), APYQuantity),
        a=convert(pot.a(0), APYQuantity),
        b=convert(pot.b(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Stone & Ostriker potentials


@dispatch  # type: ignore[misc]
def gala_to_galax(
    gala: gp.StonePotential, /
) -> gpx.StoneOstriker15Potential | gpx.PotentialFrame:
    """Convert a `gala.potential.StonePotential` to a `galax.potential.StoneOstriker15Potential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.StonePotential(m=1e11, r_c=20, r_h=10, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    StoneOstriker15Potential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      r_c=ConstantParameter( ... ),
      r_h=ConstantParameter( ... )
    )
    """  # noqa: E501
    params = gala.parameters
    pot = gpx.StoneOstriker15Potential(
        m_tot=params["m"], r_c=params["r_c"], r_h=params["r_h"], units=gala.units
    )
    return _apply_frame(_get_frame(gala), pot)


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.StoneOstriker15Potential, /) -> gp.StonePotential:
    """Convert a `galax.potential.StoneOstriker15Potential` to a `gala.potential.StonePotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.StoneOstriker15Potential(m_tot=Quantity(1e11, "Msun"), r_c=Quantity(20, "kpc"), r_h=Quantity(10, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <StonePotential: m=1.00e+11, r_c=20.00, r_h=10.00 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return gp.StonePotential(
        m=convert(pot.m_tot(0), APYQuantity),
        r_c=convert(pot.r_c(0), APYQuantity),
        r_h=convert(pot.r_h(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# -----------------------------------------------------------------------------
# Logarithmic potentials


@dispatch  # type: ignore[misc]
def gala_to_galax(
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
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.LogarithmicPotential(v_c=220, r_h=20, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    LogarithmicPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
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


@dispatch
def galax_to_gala(pot: gpx.LogarithmicPotential, /) -> gp.LogarithmicPotential:
    """Convert a `galax.potential.LogarithmicPotential` to a `gala.potential.LogarithmicPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.LogarithmicPotential(v_c=Quantity(220, "km/s"), r_s=Quantity(20, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <LogarithmicPotential: v_c=0.22, r_h=20.00, q1=1.00, q2=1.00, q3=1.00, phi=0 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return gp.LogarithmicPotential(
        v_c=convert(pot.v_c(0), APYQuantity),
        r_h=convert(pot.r_s(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


@dispatch
def galax_to_gala(pot: gpx.LMJ09LogarithmicPotential, /) -> gp.LogarithmicPotential:
    """Convert a `galax.potential.LMJ09LogarithmicPotential` to a `gala.potential.LogarithmicPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.LMJ09LogarithmicPotential(
    ...     v_c=Quantity(220, "km/s"),
    ...     r_s=Quantity(20, "kpc"),
    ...     q1=1.0, q2=1.0, q3=1.0,
    ...     phi=Quantity(0, "rad"),
    ...     units="galactic",
    ... )
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <LogarithmicPotential: v_c=0.22, r_h=20.00, q1=1.00, q2=1.00, q3=1.00, phi=0 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
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
# Multipole potentials


@dispatch  # type: ignore[misc]
def gala_to_galax(
    gala: gp.MultipolePotential, /
) -> gpx.MultipoleInnerPotential | gpx.MultipoleOuterPotential | gpx.PotentialFrame:
    params = gala.parameters
    cls = (
        gpx.MultipoleInnerPotential
        if params["inner"] == 1
        else gpx.MultipoleOuterPotential
    )

    l_max = gala._lmax  # noqa: SLF001
    Slm = jnp.zeros((l_max + 1, l_max + 1), dtype=float)
    Tlm = jnp.zeros_like(Slm)

    for l, m in zip(*jnp.tril_indices(l_max + 1), strict=True):
        skey = f"S{l}{m}"
        if skey in params:
            Slm = Slm.at[l, m].set(params[skey])

        tkey = f"T{l}{m}"
        if tkey in params:
            Tlm = Tlm.at[l, m].set(params[tkey])

    pot = cls(
        m_tot=params["m"],
        r_s=params["r_s"],
        l_max=l_max,
        Slm=Slm,
        Tlm=Tlm,
        units=gala.units,
    )
    return _apply_frame(_get_frame(gala), pot)


@dispatch.multi((gpx.MultipoleInnerPotential,), (gpx.MultipoleOuterPotential,))  # type: ignore[misc]
def galax_to_gala(
    pot: gpx.MultipoleInnerPotential | gpx.MultipoleOuterPotential, /
) -> gp.MultipolePotential:
    """Convert a Galax Multipole to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, "m_tot", "r_s", "Slm", "Tlm")

    Slm, Tlm = pot.Slm(0).value, pot.Tlm(0).value
    ls, ms = jnp.tril_indices(pot.l_max + 1)

    return gp.MultipolePotential(
        m=convert(pot.m_tot(0), APYQuantity),
        r_s=convert(pot.r_s(0), APYQuantity),
        lmax=pot.l_max,
        **{
            f"S{l}{m}": Slm[l, m] for l, m in zip(ls, ms, strict=True) if Slm[l, m] != 0
        },
        **{
            f"T{l}{m}": Tlm[l, m] for l, m in zip(ls, ms, strict=True) if Tlm[l, m] != 0
        },
        inner=isinstance(pot, gpx.MultipoleInnerPotential),
        units=_galax_to_gala_units(pot.units),
    )


# -----------------------------------------------------------------------------
# NFW potentials


@dispatch
def gala_to_galax(gala: gp.NFWPotential, /) -> gpx.NFWPotential | gpx.PotentialFrame:
    """Convert a Gala NFWPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import gala.units as gu
    >>> import galax.potential as gpx

    >>> gpot = gp.NFWPotential(m=1e12, r_s=20, units=gu.galactic)
    >>> gpx.io.convert_potential(gpx.io.GalaxLibrary, gpot)
    NFWPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m=ConstantParameter( ... ),
      r_s=ConstantParameter( ... )
    )

    """
    params = gala.parameters
    pot = gpx.NFWPotential(m=params["m"], r_s=params["r_s"], units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def gala_to_galax(
    pot: gp.LeeSutoTriaxialNFWPotential, /
) -> gpx.LeeSutoTriaxialNFWPotential:
    """Convert a :class:`gala.potential.LeeSutoTriaxialNFWPotential` to a :class:`galax.potential.LeeSutoTriaxialNFWPotential`.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import gala.units as gu
    >>> import galax.potential as gpx

    >>> gpot = gp.LeeSutoTriaxialNFWPotential(
    ...     v_c=220, r_s=20, a=1, b=0.9, c=0.8, units=gu.galactic )
    >>> gpx.io.convert_potential(gpx.io.GalaxLibrary, gpot)
    LeeSutoTriaxialNFWPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m=ConstantParameter( ... ),
      r_s=ConstantParameter( ... ),
      a1=ConstantParameter( ... ),
      a2=ConstantParameter( ... ),
      a3=ConstantParameter( ... )
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


# -----------------------------------------------------------------------------
# NFW potentials


@dispatch
def gala_to_galax(gala: gp.NFWPotential, /) -> gpx.NFWPotential | gpx.PotentialFrame:
    """Convert a Gala NFWPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.NFWPotential(m=1e12, r_s=20, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    NFWPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m=ConstantParameter( value=Quantity[...](value=f64[], unit=Unit("solMass")) ),
      r_s=ConstantParameter( value=Quantity[...](value=f64[], unit=Unit("kpc")) )
    )

    """
    params = gala.parameters
    pot = gpx.NFWPotential(
        m=params["m"], r_s=params["r_s"], units=_check_gala_units(gala.units)
    )
    return _apply_frame(_get_frame(gala), pot)


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.NFWPotential, /) -> gp.NFWPotential:
    """Convert a `galax.potential.NFWPotential` to a `gala.potential.NFWPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.NFWPotential(m=Quantity(1e12, "Msun"), r_s=Quantity(20, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <NFWPotential: m=1.00e+12, r_s=20.00, a=1.00, b=1.00, c=1.00 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return gp.NFWPotential(
        m=convert(pot.m(0), APYQuantity),
        r_s=convert(pot.r_s(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


@dispatch  # type: ignore[misc]
def gala_to_galax(
    pot: gp.LeeSutoTriaxialNFWPotential, /
) -> gpx.LeeSutoTriaxialNFWPotential:
    """Convert a `gala.potential.LeeSutoTriaxialNFWPotential` to a `galax.potential.LeeSutoTriaxialNFWPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.LeeSutoTriaxialNFWPotential(
    ...     v_c=220, r_s=20, a=1, b=0.9, c=0.8, units=galactic )
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    LeeSutoTriaxialNFWPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m=ConstantParameter( ... ),
      r_s=ConstantParameter( ... ),
      a1=ConstantParameter( ... ),
      a2=ConstantParameter( ... ),
      a3=ConstantParameter( ... )
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


@dispatch  # type: ignore[misc]
def galax_to_gala(
    pot: gpx.LeeSutoTriaxialNFWPotential, /
) -> gp.LeeSutoTriaxialNFWPotential:
    """Convert a `galax.potential.LeeSutoTriaxialNFWPotential` to a `gala.potential.LeeSutoTriaxialNFWPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.LeeSutoTriaxialNFWPotential(
    ...     m=Quantity(1e12, "Msun"),
    ...     r_s=Quantity(20, "kpc"),
    ...     a1=Quantity(1, ""),
    ...     a2=Quantity(0.9, ""),
    ...     a3=Quantity(0.8, ""),
    ...     units="galactic",
    ... )
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <LeeSutoTriaxialNFWPotential: v_c=0.47, r_s=20.00, a=1.00, b=0.90, c=0.80 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    t = Quantity(0.0, pot.units["time"])
    v_c = convert(xp.sqrt(pot.constants["G"] * pot.m(t) / pot.r_s(t)), APYQuantity)

    return gp.LeeSutoTriaxialNFWPotential(
        v_c=v_c,
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


@dispatch  # type: ignore[misc]
def gala_to_galax(pot: gp.BovyMWPotential2014, /) -> gpx.BovyMWPotential2014:
    """Convert a Gala BovyMWPotential2014 to a Galax potential.

    Examples
    --------
    .. invisible-code-block: python

        from galax._interop.optional_deps import GSL_ENABLED

    .. skip: start if(not GSL_ENABLED, reason="requires GSL")

    >>> import gala.potential as galap
    >>> import galax.potential as gp

    >>> pot = galap.BovyMWPotential2014()
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
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


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.BovyMWPotential2014, /) -> gp.BovyMWPotential2014:
    """Convert a `gala.potential.BovyMWPotential2014` to a `galax.potential.BovyMWPotential2014`.

    Examples
    --------
    .. invisible-code-block: python

        from galax._interop.optional_deps import GSL_ENABLED

    .. skip: start if(not GSL_ENABLED, reason="requires GSL")

    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.BovyMWPotential2014(
    ...     disk=gp.MiyamotoNagaiPotential(m_tot=Quantity(1e11, "Msun"), a=Quantity(6.5, "kpc"), b=Quantity(0.26, "kpc"), units="galactic"),
    ...     bulge=gp.PowerLawCutoffPotential(m_tot=Quantity(1e10, "Msun"), alpha=1.8, r_c=Quantity(20, "kpc"), units="galactic"),
    ...     halo=gp.NFWPotential(m=Quantity(1e12, "Msun"), r_s=Quantity(20, "kpc"), units="galactic"),
    ... )
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <CompositePotential disk,bulge,halo>

    """  # noqa: E501

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


@dispatch  # type: ignore[misc]
def gala_to_galax(pot: gp.LM10Potential, /) -> gpx.LM10Potential:
    """Convert a `gala.potential.LM10Potential` to a `galax.potential.LM10Potential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> import galax.potential as gp

    >>> pot = galap.LM10Potential()
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    LM10Potential({'disk': MiyamotoNagaiPotential( ... ),
                   'bulge': HernquistPotential( ... ),
                   'halo': LMJ09LogarithmicPotential( ... )})

    """
    return gpx.LM10Potential(
        disk=gala_to_galax(pot["disk"]),
        bulge=gala_to_galax(pot["bulge"]),
        halo=gala_to_galax(pot["halo"]),
    )


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.LM10Potential, /) -> gp.LM10Potential:
    """Convert a `galax.potential.LM10Potential` to a `gala.potential.LM10Potential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.LM10Potential(
    ...     disk=gp.MiyamotoNagaiPotential(m_tot=Quantity(1e11, "Msun"), a=Quantity(6.5, "kpc"), b=Quantity(0.26, "kpc"), units="galactic"),
    ...     bulge=gp.HernquistPotential(m_tot=Quantity(1e10, "Msun"), r_s=Quantity(1, "kpc"), units="galactic"),
    ...     halo=gp.LMJ09LogarithmicPotential(v_c=Quantity(220, "km/s"), r_s=Quantity(20, "kpc"), q1=1, q2=1, q3=1, phi=Quantity(0, "rad"), units="galactic"),
    ... )

    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <CompositePotential disk,bulge,halo>

    """  # noqa: E501

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


@dispatch  # type: ignore[misc]
def gala_to_galax(pot: gp.MilkyWayPotential, /) -> gpx.MilkyWayPotential:
    """Convert a `gala.potential.MilkyWayPotential` to a `galax.potential.MilkyWayPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> import galax.potential as gp

    >>> pot = galap.MilkyWayPotential()
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    MilkyWayPotential({'disk': MiyamotoNagaiPotential( ... ),
                       'halo': NFWPotential( ... ),
                       'bulge': HernquistPotential( ... ),
                       'nucleus': HernquistPotential( ... )})

    """  # noqa: E501
    return gpx.MilkyWayPotential(
        disk=gala_to_galax(pot["disk"]),
        halo=gala_to_galax(pot["halo"]),
        bulge=gala_to_galax(pot["bulge"]),
        nucleus=gala_to_galax(pot["nucleus"]),
    )


@dispatch  # type: ignore[misc]
def galax_to_gala(pot: gpx.MilkyWayPotential, /) -> gp.MilkyWayPotential:
    """Convert a `galax.potential.MilkyWayPotential` to a `gala.potential.MilkyWayPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.MilkyWayPotential(
    ...     disk=dict(m_tot=Quantity(1e11, "Msun"), a=Quantity(6.5, "kpc"), b=Quantity(0.26, "kpc")),
    ...     halo=dict(m=Quantity(1e12, "Msun"), r_s=Quantity(20, "kpc")),
    ...     bulge=dict(m_tot=Quantity(1e10, "Msun"), r_s=Quantity(1, "kpc")),
    ...     nucleus=dict(m_tot=Quantity(1e9, "Msun"), r_s=Quantity(0.1, "kpc")),
    ... )

    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <CompositePotential disk,bulge,nucleus,halo>

    """  # noqa: E501

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

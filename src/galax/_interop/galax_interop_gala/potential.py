"""Interoperability with :mod:`gala` potentials."""

__all__ = ["gala_to_galax", "galax_to_gala"]

from typing import TypeVar

import equinox as eqx
import gala.potential as galap
from astropy.units import Quantity as APYQuantity
from gala.units import (
    DimensionlessUnitSystem as GalaDimensionlessUnitSystem,
    UnitSystem as GalaUnitSystem,
    dimensionless as gala_dimensionless,
)
from packaging.version import Version
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.unitsystems import AbstractUnitSystem, DimensionlessUnitSystem

import galax.potential as gp
import galax.potential.io as gpio
from galax._interop.optional_deps import OptDeps

##############################################################################
# Hook into general dispatcher


@dispatch
def convert_potential(
    to_: gp.AbstractPotential | type[gpio.GalaxLibrary],  # noqa: ARG001
    from_: galap.CPotentialBase | galap.PotentialBase,
    /,
) -> gp.AbstractPotential:
    """Convert a :class:`~gala.potential.PotentialBase` to a :class:`~galax.potential.AbstractPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.KeplerPotential(m=1e11, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    KeplerPotential(
      units=LTMAUnitSystem( ... ),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( value=Quantity[...](value=f64[], unit=Unit("solMass")) )
    )

    """  # noqa: E501
    return gala_to_galax(from_)


@dispatch
def convert_potential(
    to_: galap.CPotentialBase | galap.PotentialBase | type[gpio.GalaLibrary],  # noqa: ARG001
    from_: gp.AbstractPotential,
    /,
) -> galap.CPotentialBase | galap.PotentialBase:
    """Convert a :class:`~galax.potential.AbstractPotential` to a :class:`~gala.potential.PotentialBase`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e11, "Msun"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <KeplerPotential: m=1.00e+11 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    return galax_to_gala(from_)


# NOTE: this is a bit of type piracy, but since `gala` does not use `plum` and
# so does not support this function, this is totally fine.
@dispatch
def convert_potential(
    to_: galap.CPotentialBase | galap.PotentialBase | type[gpio.GalaLibrary],  # noqa: ARG001
    from_: galap.CPotentialBase | galap.PotentialBase,
    /,
) -> galap.CPotentialBase | galap.PotentialBase:
    """Convert a :class:`~galax.potential.AbstractPotential` to itself.

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

PT = TypeVar("PT", bound=gp.AbstractPotential)


def _get_frame(pot: galap.PotentialBase, /) -> cx.ops.AbstractOperator:
    """Convert a Gala frame to a Galax frame."""
    frame = cx.ops.GalileanSpatialTranslation(
        u.Quantity(pot.origin, unit=pot.units["length"])
    )
    if pot.R is not None:
        frame = cx.ops.GalileanRotation(pot.R) | frame
    return cx.ops.simplify_op(frame)


def _apply_frame(frame: cx.ops.AbstractOperator, pot: PT, /) -> PT | gp.PotentialFrame:
    """Apply a Galax frame to a potential."""
    # A framed Galax potential never simplifies to a frameless potential. This
    # function applies a frame if it is not the identity operator.
    return pot if isinstance(frame, cx.ops.Identity) else gp.PotentialFrame(pot, frame)


def _galax_to_gala_units(units: AbstractUnitSystem, /) -> GalaUnitSystem:
    """Convert a Galax unit system to a Gala unit system."""
    # Galax potentials naturally convert Gala unit systems, but Gala potentials
    # do not convert Galax unit systems. This function is used for that purpose.
    if isinstance(units, DimensionlessUnitSystem):
        return gala_dimensionless
    return GalaUnitSystem(units)


def _error_if_not_all_constant_parameters(
    pot: gp.AbstractPotential, *params: str
) -> None:
    """Check if all parameters are constant."""
    is_time_dep = any(
        not isinstance(getattr(pot, name), gp.params.ConstantParameter)
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


@dispatch
def gala_to_galax(pot: galap.PotentialBase, /) -> gp.AbstractPotential:
    """Convert a :mod:`gala` potential to a :mod:`galax` potential.

    Parameters
    ----------
    pot :  :class:`~gala.potential.PotentialBase`
        :mod:`gala` potential.

    Returns
    -------
    gala_pot : :class:`~galax.potential.AbstractPotential`
        :mod:`galax` potential.

    """
    msg = (
        "`gala_to_galax` does not have a registered function to convert "
        f"{pot.__class__.__name__!r} to a `galax.AbstractPotential` instance."
    )
    raise NotImplementedError(msg)


# TODO: add an argument to specify how to handle time-dependent parameters.
#       Gala potentials are not time-dependent, so we need to specify how to
#       handle time-dependent Galax parameters.
@dispatch
def galax_to_gala(pot: gp.AbstractPotential, /) -> galap.PotentialBase:
    """Convert a Galax potential to a Gala potential.

    Parameters
    ----------
    pot : :class:`~galax.potential.AbstractPotential`
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


@dispatch
def gala_to_galax(pot: galap.CompositePotential, /) -> gp.CompositePotential:
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
    return gp.CompositePotential(**{k: gala_to_galax(p) for k, p in pot.items()})


@dispatch
def galax_to_gala(pot: gp.CompositePotential, /) -> galap.CompositePotential:
    """Convert a `galax.potential.CompositePotential` -> `gala.potential.CompositePotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.CompositePotential(
    ...     disk=gp.MiyamotoNagaiPotential(m_tot=u.Quantity(1e11, "Msun"), a=6.5, b=0.26, units="galactic"),
    ...     halo=gp.NFWPotential(m=u.Quantity(1e12, "Msun"), r_s=20, units="galactic"),
    ... )
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <CompositePotential disk,halo>

    """  # noqa: E501
    return galap.CompositePotential(**{k: galax_to_gala(p) for k, p in pot.items()})


# -----------------------------------------------------------------------------
# Builtin potentials


@dispatch
def galax_to_gala(_: gp.BarPotential, /) -> galap.PotentialBase:
    """Convert a Galax BarPotential to a Gala potential."""
    raise NotImplementedError  # TODO: implement


if OptDeps.GALA.installed and (Version("1.8.2") <= OptDeps.GALA):

    @dispatch
    def gala_to_galax(
        gala: galap.BurkertPotential, /
    ) -> gp.BurkertPotential | gp.PotentialFrame:
        """Convert a `gala.potential.BurkertPotential` to a galax.potential.BurkertPotential.

        Examples
        --------
        >>> import gala.potential as galap
        >>> from gala.units import galactic
        >>> import galax.potential as gp

        .. invisible-code-block: python

            from packaging.version import Version
            from galax._interop.optional_deps import OptDeps
            skip = not OptDeps.GALA.installed or OptDeps.GALA < Version("1.8.2")

        .. skip: start if(skip, reason="Requires Gala v1.8.2+")

        >>> pot = galap.BurkertPotential(rho=4, r0=20, units=galactic)
        >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
        BurkertPotential(
        units=LTMAUnitSystem( length=Unit("kpc"), ...),
        constants=ImmutableMap({'G': ...}),
        m=ConstantParameter( ... ),
        r_s=ConstantParameter( ... )
        )

        .. skip: end

        """  # noqa: E501
        params = gala.parameters
        pot = gp.BurkertPotential.from_central_density(
            rho_0=params["rho"], r_s=params["r0"], units=gala.units
        )
        return _apply_frame(_get_frame(gala), pot)

    @dispatch
    def galax_to_gala(pot: gp.BurkertPotential, /) -> galap.BurkertPotential:
        """Convert a `galax.potential.BurkertPotential` to a `gala.potential.BurkertPotential`.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        .. invisible-code-block: python

            from packaging.version import Version
            from galax._interop.optional_deps import OptDeps
            skip = not OptDeps.GALA.installed or OptDeps.GALA < Version("1.8.2")

        .. skip: start if(skip, reason="Requires Gala v1.8.2+")

        >>> pot = gp.BurkertPotential(m=u.Quantity(1e11, "Msun"), r_s=u.Quantity(20, "kpc"), units="galactic")
        >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
        <BurkertPotential: rho=7.82e+06, r0=20.00 (kpc,Myr,solMass,rad)>

        .. skip: end

        """  # noqa: E501
        _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

        return galap.BurkertPotential(
            rho=convert(pot.rho0(0), APYQuantity),
            r0=convert(pot.r_s(0), APYQuantity),
            units=_galax_to_gala_units(pot.units),
        )

# ---------------------------
# Harmonic oscillator potentials


@dispatch
def gala_to_galax(
    gala: galap.HarmonicOscillatorPotential, /
) -> gp.HarmonicOscillatorPotential | gp.PotentialFrame:
    r"""Convert a `gala.potential.HarmonicOscillatorPotential` to a `galax.potential.HarmonicOscillatorPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.HarmonicOscillatorPotential(omega=1, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    HarmonicOscillatorPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      omega=ConstantParameter( ... )
    )

    """  # noqa: E501
    params = gala.parameters
    pot = gp.HarmonicOscillatorPotential(
        omega=params["omega"], units=_check_gala_units(gala.units)
    )
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(
    pot: gp.HarmonicOscillatorPotential, /
) -> galap.HarmonicOscillatorPotential:
    """Convert a `galax.potential.HarmonicOscillatorPotential` to a `gala.potential.HarmonicOscillatorPotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.HarmonicOscillatorPotential(omega=u.Quantity(1, "1/Myr"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <HarmonicOscillatorPotential: omega=[1.] (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return galap.HarmonicOscillatorPotential(
        omega=convert(pot.omega(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Hernquist potentials


@dispatch
def gala_to_galax(
    gala: galap.HernquistPotential, /
) -> gp.HernquistPotential | gp.PotentialFrame:
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
    pot = gp.HernquistPotential(
        m_tot=params["m"], r_s=params["c"], units=_check_gala_units(gala.units)
    )
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(pot: gp.HernquistPotential, /) -> galap.HernquistPotential:
    """Convert a `galax.potential.HernquistPotential` to a `gala.potential.HernquistPotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e11, "Msun"), r_s=u.Quantity(20, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <HernquistPotential: m=1.00e+11, c=20.00 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return galap.HernquistPotential(
        m=convert(pot.m_tot(0), APYQuantity),
        c=convert(pot.r_s(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Isochrone potentials


@dispatch
def gala_to_galax(
    gala: galap.IsochronePotential, /
) -> gp.IsochronePotential | gp.PotentialFrame:
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

    pot = gp.IsochronePotential(**params, units=_check_gala_units(gala.units))
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(pot: gp.IsochronePotential, /) -> galap.IsochronePotential:
    """Convert a `galax.potential.IsochronePotential` to a `gala.potential.IsochronePotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.IsochronePotential(m_tot=u.Quantity(1e11, "Msun"), b=u.Quantity(10, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <IsochronePotential: m=1.00e+11, b=10.00 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    params: dict[str, APYQuantity] = {
        k: convert(getattr(pot, k)(0), APYQuantity)
        for (k, f) in type(pot).parameters.items()
    }
    if "m_tot" in params:
        params["m"] = params.pop("m_tot")

    return galap.IsochronePotential(**params, units=_galax_to_gala_units(pot.units))


# ---------------------------
# Jaffe potentials


@dispatch
def gala_to_galax(
    gala: galap.JaffePotential, /
) -> gp.JaffePotential | gp.PotentialFrame:
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
    pot = gp.JaffePotential(
        m=params["m"], r_s=params["c"], units=_check_gala_units(gala.units)
    )
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(pot: gp.JaffePotential, /) -> galap.JaffePotential:
    """Convert a `galax.potential.JaffePotential` to a `gala.potential.JaffePotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.JaffePotential(m=u.Quantity(1e11, "Msun"), r_s=u.Quantity(20, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <JaffePotential: m=1.00e+11, c=20.00 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return galap.JaffePotential(
        m=convert(pot.m(0), APYQuantity),
        c=convert(pot.r_s(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Kepler potentials


@dispatch
def gala_to_galax(
    gala: galap.KeplerPotential, /
) -> gp.KeplerPotential | gp.PotentialFrame:
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

    pot = gp.KeplerPotential(**params, units=_check_gala_units(gala.units))
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(pot: gp.KeplerPotential, /) -> galap.KeplerPotential:
    """Convert a `galax.potential.KeplerPotential` to a `gala.potential.KeplerPotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e11, "Msun"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <KeplerPotential: m=1.00e+11 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    params: dict[str, APYQuantity] = {
        k: convert(getattr(pot, k)(0), APYQuantity)
        for (k, f) in type(pot).parameters.items()
    }
    if "m_tot" in params:
        params["m"] = params.pop("m_tot")

    return galap.KeplerPotential(**params, units=_galax_to_gala_units(pot.units))


# ---------------------------
# Kuzmin potentials


@dispatch
def gala_to_galax(
    gala: galap.KuzminPotential, /
) -> gp.KuzminPotential | gp.PotentialFrame:
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

    pot = gp.KuzminPotential(**params, units=_check_gala_units(gala.units))
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(pot: gp.KuzminPotential, /) -> galap.KuzminPotential:
    """Convert a `galax.potential.KuzminPotential` to a `gala.potential.KuzminPotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.KuzminPotential(m_tot=u.Quantity(1e11, "Msun"), a=u.Quantity(20, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <KuzminPotential: m=1.00e+11, a=20.00 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    params: dict[str, APYQuantity] = {
        k: convert(getattr(pot, k)(0), APYQuantity)
        for (k, f) in type(pot).parameters.items()
    }
    if "m_tot" in params:
        params["m"] = params.pop("m_tot")

    return galap.KuzminPotential(**params, units=_galax_to_gala_units(pot.units))


# ---------------------------
# Long & Murali Bar potentials


@dispatch
def gala_to_galax(
    gala: galap.LongMuraliBarPotential, /
) -> gp.LongMuraliBarPotential | gp.PotentialFrame:
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
    pot = gp.LongMuraliBarPotential(
        m_tot=params["m"],
        a=params["a"],
        b=params["b"],
        c=params["c"],
        alpha=params["alpha"],
        units=gala.units,
    )
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(pot: gp.LongMuraliBarPotential, /) -> galap.LongMuraliBarPotential:
    """Convert a `galax.potential.LongMuraliBarPotential` to a `gala.potential.LongMuraliBarPotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.LongMuraliBarPotential(
    ...     m_tot=u.Quantity(1e11, "Msun"),
    ...     a=u.Quantity(20, "kpc"),
    ...     b=u.Quantity(10, "kpc"),
    ...     c=u.Quantity(5, "kpc"),
    ...     alpha=u.Quantity(0.1, "rad"),
    ...     units="galactic",
    ... )
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <LongMuraliBarPotential: m=1.00e+11, a=20.00, b=10.00, c=5.00, alpha=0.10 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return galap.LongMuraliBarPotential(
        m=convert(pot.m_tot(0), APYQuantity),
        a=convert(pot.a(0), APYQuantity),
        b=convert(pot.b(0), APYQuantity),
        c=convert(pot.c(0), APYQuantity),
        alpha=convert(pot.alpha(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Miyamoto-Nagai potentials


@dispatch
def gala_to_galax(
    gala: galap.MiyamotoNagaiPotential, /
) -> gp.MiyamotoNagaiPotential | gp.PotentialFrame:
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

    pot = gp.MiyamotoNagaiPotential(**params, units=_check_gala_units(gala.units))
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(pot: gp.MiyamotoNagaiPotential, /) -> galap.MiyamotoNagaiPotential:
    """Convert a `galax.potential.MiyamotoNagaiPotential` to a `gala.potential.MiyamotoNagaiPotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.MiyamotoNagaiPotential(m_tot=u.Quantity(1e11, "Msun"), a=u.Quantity(6.5, "kpc"), b=u.Quantity(0.26, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <MiyamotoNagaiPotential: m=1.00e+11, a=6.50, b=0.26 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    params: dict[str, APYQuantity] = {
        k: convert(getattr(pot, k)(0), APYQuantity)
        for (k, f) in type(pot).parameters.items()
    }
    if "m_tot" in params:
        params["m"] = params.pop("m_tot")

    return galap.MiyamotoNagaiPotential(**params, units=_galax_to_gala_units(pot.units))


# ---------------------------
# MN3 potentials


@dispatch
def gala_to_galax(
    gala: galap.MN3ExponentialDiskPotential, /
) -> gp.MN3ExponentialPotential | gp.MN3Sech2Potential | gp.PotentialFrame:
    """Convert a `gala.potential.MN3ExponentialDiskPotential` to a `galax.potential.MN3ExponentialPotential` or `galax.potential.MN3Sech2Potential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> pot = galap.MN3ExponentialDiskPotential(m=1e11, h_R=3., h_z=0.2, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    MN3Sech2Potential(
      units=LTMAUnitSystem( ... ),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      h_R=ConstantParameter( ... ),
      h_z=ConstantParameter( ... ),
      positive_density=True
    )

    """  # noqa: E501
    params = dict(gala.parameters)
    params["m_tot"] = params.pop("m")
    params["positive_density"] = gala.positive_density

    cls = gp.MN3Sech2Potential if gala.sech2_z else gp.MN3ExponentialPotential

    pot = cls(**params, units=_check_gala_units(gala.units))
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(
    pot: gp.MN3ExponentialPotential | gp.MN3Sech2Potential, /
) -> galap.MN3ExponentialDiskPotential:
    """Convert a `galax.potential.MN3ExponentialPotential` or `galax.potential.MN3Sech2Potential` to a `gala.potential.MN3ExponentialDiskPotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.MN3ExponentialPotential(m_tot=u.Quantity(1e11, "Msun"), h_R=u.Quantity(3.0, "kpc"), h_z=u.Quantity(0.2, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <MN3ExponentialDiskPotential: m=1.00e+11, h_R=3.00, h_z=0.20 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    params: dict[str, APYQuantity] = {
        k: convert(getattr(pot, k)(0), APYQuantity)
        for (k, f) in type(pot).parameters.items()
    }
    if "m_tot" in params:
        params["m"] = params.pop("m_tot")

    params["sech2_z"] = isinstance(pot, gp.MN3Sech2Potential)
    params["positive_density"] = pot.positive_density

    return galap.MN3ExponentialDiskPotential(
        **params, units=_galax_to_gala_units(pot.units)
    )


# ---------------------------
# Null potentials


@dispatch
def gala_to_galax(pot: galap.NullPotential, /) -> gp.NullPotential:
    """Convert a `gala.potential.NullPotential` to a `galax.potential.NullPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> import galax.potential as gp
    >>> from gala.units import galactic

    >>> pot = galap.NullPotential(units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    NullPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...})
    )

    """
    return gp.NullPotential(units=pot.units)


@dispatch
def galax_to_gala(pot: gp.NullPotential, /) -> galap.NullPotential:
    """Convert a `galax.potential.NullPotential` to a `gala.potential.NullPotential`.

    Examples
    --------
    >>> import galax.potential as gp

    >>> pot = gp.NullPotential()
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <NullPotential:  (kpc,Myr,solMass,rad)>

    """
    return galap.NullPotential(
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Plummer potentials


@dispatch
def gala_to_galax(
    gala: galap.PlummerPotential, /
) -> gp.PlummerPotential | gp.PotentialFrame:
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

    pot = gp.PlummerPotential(**params, units=_check_gala_units(gala.units))
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(pot: gp.PlummerPotential, /) -> galap.PlummerPotential:
    """Convert a `galax.potential.PlummerPotential` to a `gala.potential.PlummerPotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.PlummerPotential(m_tot=u.Quantity(1e11, "Msun"), b=u.Quantity(10, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <PlummerPotential: m=1.00e+11, b=10.00 (kpc,Myr,solMass,rad)>
    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    params: dict[str, APYQuantity] = {
        k: convert(getattr(pot, k)(0), APYQuantity)
        for (k, f) in type(pot).parameters.items()
    }
    if "m_tot" in params:
        params["m"] = params.pop("m_tot")

    return galap.PlummerPotential(**params, units=_galax_to_gala_units(pot.units))


# ---------------------------
# PowerLawCutoff potentials


@dispatch
def gala_to_galax(
    gala: galap.PowerLawCutoffPotential, /
) -> gp.PowerLawCutoffPotential | gp.PotentialFrame:
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

    pot = gp.PowerLawCutoffPotential(**params, units=_check_gala_units(gala.units))
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(pot: gp.PowerLawCutoffPotential, /) -> galap.PowerLawCutoffPotential:
    """Convert a `galax.potential.PowerLawCutoffPotential` to a `gala.potential.PowerLawCutoffPotential`.

    Examples
    --------
    .. invisible-code-block: python

        from galax._interop.optional_deps import GSL_ENABLED

    .. skip: start if(not GSL_ENABLED, reason="requires GSL")

    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.PowerLawCutoffPotential(m_tot=u.Quantity(1e11, "Msun"), alpha=1.8, r_c=u.Quantity(20, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <PowerLawCutoffPotential: m=1.00e+11, alpha=1.80, r_c=20.00 (kpc,Myr,solMass,rad)>

    .. skip: end
    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    params: dict[str, APYQuantity] = {
        k: convert(getattr(pot, k)(0), APYQuantity)
        for (k, f) in type(pot).parameters.items()
    }
    if "m_tot" in params:
        params["m"] = params.pop("m_tot")

    return galap.PowerLawCutoffPotential(
        **params, units=_galax_to_gala_units(pot.units)
    )


# ---------------------------
# Satoh potentials


@dispatch
def gala_to_galax(
    gala: galap.SatohPotential, /
) -> gp.SatohPotential | gp.PotentialFrame:
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
    pot = gp.SatohPotential(
        m_tot=params["m"], a=params["a"], b=params["b"], units=gala.units
    )
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(pot: gp.SatohPotential, /) -> galap.SatohPotential:
    """Convert a `galax.potential.SatohPotential` to a `gala.potential.SatohPotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.SatohPotential(m_tot=u.Quantity(1e11, "Msun"), a=u.Quantity(20, "kpc"), b=u.Quantity(10, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <SatohPotential: m=1.00e+11, a=20.00, b=10.00 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return galap.SatohPotential(
        m=convert(pot.m_tot(0), APYQuantity),
        a=convert(pot.a(0), APYQuantity),
        b=convert(pot.b(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# ---------------------------
# Stone & Ostriker potentials


@dispatch
def gala_to_galax(
    gala: galap.StonePotential, /
) -> gp.StoneOstriker15Potential | gp.PotentialFrame:
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
    pot = gp.StoneOstriker15Potential(
        m_tot=params["m"], r_c=params["r_c"], r_h=params["r_h"], units=gala.units
    )
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(pot: gp.StoneOstriker15Potential, /) -> galap.StonePotential:
    """Convert a `galax.potential.StoneOstriker15Potential` to a `gala.potential.StonePotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.StoneOstriker15Potential(m_tot=u.Quantity(1e11, "Msun"), r_c=u.Quantity(20, "kpc"), r_h=u.Quantity(10, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <StonePotential: m=1.00e+11, r_c=20.00, r_h=10.00 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return galap.StonePotential(
        m=convert(pot.m_tot(0), APYQuantity),
        r_c=convert(pot.r_c(0), APYQuantity),
        r_h=convert(pot.r_h(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


# -----------------------------------------------------------------------------
# Logarithmic potentials


@dispatch
def gala_to_galax(
    gala: galap.LogarithmicPotential, /
) -> gp.LogarithmicPotential | gp.LMJ09LogarithmicPotential | gp.PotentialFrame:
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
        pot = gp.LMJ09LogarithmicPotential(
            v_c=params["v_c"],
            r_s=params["r_h"],
            q1=params["q1"],
            q2=params["q2"],
            q3=params["q3"],
            phi=params["phi"],
            units=gala.units,
        )
    else:
        pot = gp.LogarithmicPotential(
            v_c=params["v_c"], r_s=params["r_h"], units=gala.units
        )

    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(pot: gp.LogarithmicPotential, /) -> galap.LogarithmicPotential:
    """Convert a `galax.potential.LogarithmicPotential` to a `gala.potential.LogarithmicPotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.LogarithmicPotential(v_c=u.Quantity(220, "km/s"), r_s=u.Quantity(20, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <LogarithmicPotential: v_c=0.22, r_h=20.00, q1=1.00, q2=1.00, q3=1.00, phi=0 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return galap.LogarithmicPotential(
        v_c=convert(pot.v_c(0), APYQuantity),
        r_h=convert(pot.r_s(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


@dispatch
def galax_to_gala(pot: gp.LMJ09LogarithmicPotential, /) -> galap.LogarithmicPotential:
    """Convert a `galax.potential.LMJ09LogarithmicPotential` to a `gala.potential.LogarithmicPotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.LMJ09LogarithmicPotential(
    ...     v_c=u.Quantity(220, "km/s"),
    ...     r_s=u.Quantity(20, "kpc"),
    ...     q1=1.0, q2=1.0, q3=1.0,
    ...     phi=u.Quantity(0, "rad"),
    ...     units="galactic",
    ... )
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <LogarithmicPotential: v_c=0.22, r_h=20.00, q1=1.00, q2=1.00, q3=1.00, phi=0 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return galap.LogarithmicPotential(
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


@dispatch
def gala_to_galax(
    gala: galap.MultipolePotential, /
) -> gp.MultipoleInnerPotential | gp.MultipoleOuterPotential | gp.PotentialFrame:
    params = gala.parameters
    cls = (
        gp.MultipoleInnerPotential
        if params["inner"] == 1
        else gp.MultipoleOuterPotential
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


@dispatch.multi((gp.MultipoleInnerPotential,), (gp.MultipoleOuterPotential,))
def galax_to_gala(
    pot: gp.MultipoleInnerPotential | gp.MultipoleOuterPotential, /
) -> galap.MultipolePotential:
    """Convert a Galax Multipole to a Gala potential."""
    _error_if_not_all_constant_parameters(pot, "m_tot", "r_s", "Slm", "Tlm")

    Slm, Tlm = pot.Slm(0).value, pot.Tlm(0).value
    ls, ms = jnp.tril_indices(pot.l_max + 1)

    return galap.MultipolePotential(
        m=convert(pot.m_tot(0), APYQuantity),
        r_s=convert(pot.r_s(0), APYQuantity),
        lmax=pot.l_max,
        **{
            f"S{l}{m}": Slm[l, m] for l, m in zip(ls, ms, strict=True) if Slm[l, m] != 0
        },
        **{
            f"T{l}{m}": Tlm[l, m] for l, m in zip(ls, ms, strict=True) if Tlm[l, m] != 0
        },
        inner=isinstance(pot, gp.MultipoleInnerPotential),
        units=_galax_to_gala_units(pot.units),
    )


# -----------------------------------------------------------------------------
# NFW potentials


@dispatch
def gala_to_galax(gala: galap.NFWPotential, /) -> gp.NFWPotential | gp.PotentialFrame:
    """Convert a Gala NFWPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> gpot = galap.NFWPotential(m=1e12, r_s=20, units=galactic)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, gpot)
    NFWPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m=ConstantParameter( ... ),
      r_s=ConstantParameter( ... )
    )

    """
    params = gala.parameters
    pot = gp.NFWPotential(m=params["m"], r_s=params["r_s"], units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def gala_to_galax(
    pot: galap.LeeSutoTriaxialNFWPotential, /
) -> gp.LeeSutoTriaxialNFWPotential:
    """Convert a :class:`gala.potential.LeeSutoTriaxialNFWPotential` to a :class:`galax.potential.LeeSutoTriaxialNFWPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> from gala.units import galactic
    >>> import galax.potential as gp

    >>> gpot = galap.LeeSutoTriaxialNFWPotential(
    ...     v_c=220, r_s=20, a=1, b=0.9, c=0.8, units=galactic )
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, gpot)
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
    G = u.Quantity(pot.G, units["length"] ** 3 / units["time"] ** 2 / units["mass"])

    return gp.LeeSutoTriaxialNFWPotential(
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
def gala_to_galax(gala: galap.NFWPotential, /) -> gp.NFWPotential | gp.PotentialFrame:
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
    pot = gp.NFWPotential(
        m=params["m"], r_s=params["r_s"], units=_check_gala_units(gala.units)
    )
    return _apply_frame(_get_frame(gala), pot)


@dispatch
def galax_to_gala(pot: gp.NFWPotential, /) -> galap.NFWPotential:
    """Convert a `galax.potential.NFWPotential` to a `gala.potential.NFWPotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.NFWPotential(m=u.Quantity(1e12, "Msun"), r_s=u.Quantity(20, "kpc"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <NFWPotential: m=1.00e+12, r_s=20.00, a=1.00, b=1.00, c=1.00 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    return galap.NFWPotential(
        m=convert(pot.m(0), APYQuantity),
        r_s=convert(pot.r_s(0), APYQuantity),
        units=_galax_to_gala_units(pot.units),
    )


@dispatch
def gala_to_galax(
    pot: galap.LeeSutoTriaxialNFWPotential, /
) -> gp.LeeSutoTriaxialNFWPotential:
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
    G = u.Quantity(pot.G, units["length"] ** 3 / units["time"] ** 2 / units["mass"])

    return gp.LeeSutoTriaxialNFWPotential(
        m=params["v_c"] ** 2 * params["r_s"] / G,
        r_s=params["r_s"],
        a1=params["a"],
        a2=params["b"],
        a3=params["c"],
        units=units,
        constants={"G": G},
    )


@dispatch
def galax_to_gala(
    pot: gp.LeeSutoTriaxialNFWPotential, /
) -> galap.LeeSutoTriaxialNFWPotential:
    """Convert a `galax.potential.LeeSutoTriaxialNFWPotential` to a `gala.potential.LeeSutoTriaxialNFWPotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.LeeSutoTriaxialNFWPotential(
    ...     m=u.Quantity(1e12, "Msun"),
    ...     r_s=u.Quantity(20, "kpc"),
    ...     a1=u.Quantity(1, ""),
    ...     a2=u.Quantity(0.9, ""),
    ...     a3=u.Quantity(0.8, ""),
    ...     units="galactic",
    ... )
    >>> gp.io.convert_potential(gp.io.GalaLibrary, pot)
    <LeeSutoTriaxialNFWPotential: v_c=0.47, r_s=20.00, a=1.00, b=0.90, c=0.80 (kpc,Myr,solMass,rad)>

    """  # noqa: E501
    _error_if_not_all_constant_parameters(pot, *pot.parameters.keys())

    t = u.Quantity(0.0, pot.units["time"])
    v_c: APYQuantity = convert(
        jnp.sqrt(pot.constants["G"] * pot.m(t) / pot.r_s(t)), APYQuantity
    )

    return galap.LeeSutoTriaxialNFWPotential(
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


@dispatch
def gala_to_galax(pot: galap.BovyMWPotential2014, /) -> gp.BovyMWPotential2014:
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
    BovyMWPotential2014(
      disk=MiyamotoNagaiPotential( ... ),
      bulge=PowerLawCutoffPotential( ... ),
      halo=NFWPotential( ... ),
      units=..., constants=...
    )

    .. skip: end

    """
    return gp.BovyMWPotential2014(
        disk=gala_to_galax(pot["disk"]),
        bulge=gala_to_galax(pot["bulge"]),
        halo=gala_to_galax(pot["halo"]),
    )


@dispatch
def galax_to_gala(pot: gp.BovyMWPotential2014, /) -> galap.BovyMWPotential2014:
    """Convert a `gala.potential.BovyMWPotential2014` to a `galax.potential.BovyMWPotential2014`.

    Examples
    --------
    .. invisible-code-block: python

        from galax._interop.optional_deps import GSL_ENABLED

    .. skip: start if(not GSL_ENABLED, reason="requires GSL")

    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.BovyMWPotential2014(
    ...     disk=gp.MiyamotoNagaiPotential(m_tot=u.Quantity(1e11, "Msun"), a=u.Quantity(6.5, "kpc"), b=u.Quantity(0.26, "kpc"), units="galactic"),
    ...     bulge=gp.PowerLawCutoffPotential(m_tot=u.Quantity(1e10, "Msun"), alpha=1.8, r_c=u.Quantity(20, "kpc"), units="galactic"),
    ...     halo=gp.NFWPotential(m=u.Quantity(1e12, "Msun"), r_s=u.Quantity(20, "kpc"), units="galactic"),
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

    return galap.BovyMWPotential2014(
        **{
            c: {rename(k): getattr(p, k)(0) for k in p.parameters}
            for c, p in pot.items()
        }
    )


# ---------------------------
# LM10 potentials


@dispatch
def gala_to_galax(pot: galap.LM10Potential, /) -> gp.LM10Potential:
    """Convert a `gala.potential.LM10Potential` to a `galax.potential.LM10Potential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> import galax.potential as gp

    >>> pot = galap.LM10Potential()
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    LM10Potential(
      disk=MiyamotoNagaiPotential( ... ),
      bulge=HernquistPotential( ... ),
      halo=LMJ09LogarithmicPotential( ... ),
      units=..., constants=...
    )

    """
    return gp.LM10Potential(
        disk=gala_to_galax(pot["disk"]),
        bulge=gala_to_galax(pot["bulge"]),
        halo=gala_to_galax(pot["halo"]),
    )


@dispatch
def galax_to_gala(pot: gp.LM10Potential, /) -> galap.LM10Potential:
    """Convert a `galax.potential.LM10Potential` to a `gala.potential.LM10Potential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.LM10Potential(
    ...     disk=gp.MiyamotoNagaiPotential(m_tot=u.Quantity(1e11, "Msun"), a=u.Quantity(6.5, "kpc"), b=u.Quantity(0.26, "kpc"), units="galactic"),
    ...     bulge=gp.HernquistPotential(m_tot=u.Quantity(1e10, "Msun"), r_s=u.Quantity(1, "kpc"), units="galactic"),
    ...     halo=gp.LMJ09LogarithmicPotential(v_c=u.Quantity(220, "km/s"), r_s=u.Quantity(20, "kpc"), q1=1, q2=1, q3=1, phi=u.Quantity(0, "rad"), units="galactic"),
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

    return galap.LM10Potential(
        **{
            c: {rename(c, k): getattr(p, k)(0) for k in p.parameters}
            for c, p in pot.items()
        }
    )


# ---------------------------
# Galax MilkyWayPotential


@dispatch
def gala_to_galax(pot: galap.MilkyWayPotential, /) -> gp.MilkyWayPotential:
    """Convert a `gala.potential.MilkyWayPotential` to a `galax.potential.MilkyWayPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> import galax.potential as gp

    >>> pot = galap.MilkyWayPotential()
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    MilkyWayPotential(
      disk=MiyamotoNagaiPotential( ... ),
      halo=NFWPotential( ... ),
      bulge=HernquistPotential( ... ),
      nucleus=HernquistPotential( ... ),
      units=..., constants=...
    )

    """  # noqa: E501
    return gp.MilkyWayPotential(
        disk=gala_to_galax(pot["disk"]),
        halo=gala_to_galax(pot["halo"]),
        bulge=gala_to_galax(pot["bulge"]),
        nucleus=gala_to_galax(pot["nucleus"]),
    )


@dispatch
def galax_to_gala(pot: gp.MilkyWayPotential, /) -> galap.MilkyWayPotential:
    """Convert a `galax.potential.MilkyWayPotential` to a `gala.potential.MilkyWayPotential`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.MilkyWayPotential(
    ...     disk=dict(m_tot=u.Quantity(1e11, "Msun"), a=u.Quantity(6.5, "kpc"), b=u.Quantity(0.26, "kpc")),
    ...     halo=dict(m=u.Quantity(1e12, "Msun"), r_s=u.Quantity(20, "kpc")),
    ...     bulge=dict(m_tot=u.Quantity(1e10, "Msun"), r_s=u.Quantity(1, "kpc")),
    ...     nucleus=dict(m_tot=u.Quantity(1e9, "Msun"), r_s=u.Quantity(0.1, "kpc")),
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

    return galap.MilkyWayPotential(
        **{
            c: {rename(c, k): getattr(p, k)(0) for k in p.parameters}
            for c, p in pot.items()
        }
    )


# ---------------------------
# Galax MilkyWayPotential2022


@dispatch
def gala_to_galax(pot: galap.MilkyWayPotential2022, /) -> gp.MilkyWayPotential2022:
    """Convert a `gala.potential.MilkyWayPotential` to a `galax.potential.MilkyWayPotential`.

    Examples
    --------
    >>> import gala.potential as galap
    >>> import galax.potential as gp

    >>> pot = galap.MilkyWayPotential2022()
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    MilkyWayPotential2022(
      disk=MN3Sech2Potential( ... ),
      halo=NFWPotential( ... ),
      bulge=HernquistPotential( ... ),
      nucleus=HernquistPotential( ... ),
      units=..., constants=...
    )

    """  # noqa: E501
    return gp.MilkyWayPotential2022(
        disk=gala_to_galax(pot["disk"]),
        halo=gala_to_galax(pot["halo"]),
        bulge=gala_to_galax(pot["bulge"]),
        nucleus=gala_to_galax(pot["nucleus"]),
    )


@dispatch
def galax_to_gala(pot: gp.MilkyWayPotential2022, /) -> galap.MilkyWayPotential2022:
    """Convert a `galax.potential.MilkyWayPotential2022` to a `gala.potential.MilkyWayPotential2022`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.MilkyWayPotential2022(
    ...     disk=dict(m_tot=u.Quantity(1e11, "Msun"), h_R=u.Quantity(2.8, "kpc"), h_z=u.Quantity(0.25, "kpc")),
    ...     halo=dict(m=u.Quantity(1e12, "Msun"), r_s=u.Quantity(20, "kpc")),
    ...     bulge=dict(m_tot=u.Quantity(1e10, "Msun"), r_s=u.Quantity(1, "kpc")),
    ...     nucleus=dict(m_tot=u.Quantity(1e9, "Msun"), r_s=u.Quantity(0.1, "kpc")),
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

    return galap.MilkyWayPotential2022(
        **{
            c: {
                rename(c, k): convert(getattr(p, k)(0), APYQuantity)
                for k in p.parameters
            }
            for c, p in pot.items()
        }
    )

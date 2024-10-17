"""Interoperability with :mod:`galpy` potentials."""

__all__ = ["galpy_to_galax", "galax_to_galpy"]


from typing import Annotated as Antd
from typing_extensions import Doc

import equinox as eqx
import galpy.potential as gpy
import numpy as np
from astropy.units import Quantity as AstropyQuantity
from galpy.util import conversion
from plum import convert, dispatch

from unxt import Quantity

import galax.potential as gpx
import galax.potential.io as gdio

##############################################################################
# Hook into general dispatcher


@dispatch
def convert_potential(
    to_: gpx.AbstractPotentialBase | type[gdio.GalaxLibrary],  # noqa: ARG001
    from_: gpy.Potential | list[gpy.Potential],
    /,
) -> gpx.AbstractPotentialBase:
    """Convert a :class:`~galpy.potential.Potential` to a :class:`~galax.potential.AbstractPotentialBase`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> galpy_pot = gpy.KeplerPotential(amp=u.Quantity(1e11, "Msun"))
    >>> pot = gp.io.convert_potential(gp.io.GalaxLibrary, galpy_pot)
    >>> pot
    KeplerPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=...
    )
    >>> pot.m_tot(0)
    Quantity['mass'](Array(1.e+11, dtype=float64), unit='solMass')

    """  # noqa: E501
    return galpy_to_galax(from_)


@dispatch
def convert_potential(
    to_: gpy.Potential | list[gpy.Potential] | type[gdio.GalpyLibrary],  # noqa: ARG001
    from_: gpx.AbstractPotentialBase,
    /,
) -> gpy.Potential | list[gpy.Potential]:
    """Convert a :class:`~galax.potential.AbstractPotentialBase` to a :class:`~galpy.Potential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=Quantity(1e11, "Msun"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalpyLibrary, pot)
    <galpy.potential...KeplerPotential object at ...>

    """  # noqa: E501
    return galax_to_galpy(from_)


# NOTE: this is a bit of type piracy, but since `gala` does not use `plum` and
# so does not support this function, this is totally fine.
@dispatch
def convert_potential(
    to_: gpy.Potential | list[gpy.Potential] | type[gdio.GalpyLibrary],  # noqa: ARG001
    from_: gpy.Potential | list[gpy.Potential],
    /,
) -> gpy.Potential:
    """Convert a :class:`~galpy.Potential` to itself.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> pot = gpy.KeplerPotential(amp=u.Quantity(1e11, "Msun"))
    >>> gp.io.convert_potential(gp.io.GalpyLibrary, pot) is pot
    True

    """
    return from_


##############################################################################
# GALAX <-> GALPY

# -----------------------
# Helper functions


def _error_if_not_all_constant_parameters(
    pot: gpx.AbstractPotentialBase,
) -> gpx.AbstractPotentialBase:
    """Check if all parameters are constant."""
    is_time_dep = any(
        not isinstance(getattr(pot, name), gpx.params.ConstantParameter)
        for name in pot.parameters
    )
    pot: gpx.AbstractPotentialBase = eqx.error_if(
        pot, is_time_dep, "Gala does not support time-dependent parameters."
    )
    return pot


def _galpy_mass(pot: gpy.Potential, /) -> Quantity:
    """Get the total mass of a Galpy potential."""
    return Quantity.from_(
        pot._amp * conversion.mass_in_msol(pot._vo, pot._ro),  # noqa: SLF001
        "Msun",
    )


# -----------------------------------------------------------------------------


@dispatch.abstract  # type: ignore[misc]
def galpy_to_galax(
    _: Antd[gpy.Potential, Doc("The Galpy potential to convert to galax.")], /
) -> Antd[gpx.AbstractPotentialBase, Doc("The resulting Galax potential.")]:
    """Convert a :mod:`galpy` potential to a :mod:`galax` potential.

    This dispatch is for all :mod:`galpy` potentials that do not have a
    registered function to convert them to a
    :class:`~galax.potential.AbstractPotentialBase`.

    """
    raise NotImplementedError  # pragma: no cover


# TODO: add an argument to specify how to handle time-dependent parameters.
#       Gala potentials are not time-dependent, so we need to specify how to
#       handle time-dependent Galax parameters.
@dispatch.abstract  # type: ignore[misc]
def galax_to_galpy(
    _: Antd[gpx.AbstractPotentialBase, Doc("The galax potential to convert to Galpy")],
    /,
) -> Antd[gpy.Potential, Doc("The resulting Galpy potential.")]:
    """Convert a Galax potential to a Gala potential.

    This dispatch is for all :mod:`galax` potentials that do not have a
    registered function to convert them to a
    :class:`~galpy.potential.Potential`.

    """


# ---------------------------
# Composite potentials


@dispatch  # type: ignore[misc]
def galpy_to_galax(pot: list[gpy.Potential], /) -> gpx.CompositePotential:
    """Convert a list of :mod:`galpy` potentials to a `galax.potential.CompositePotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> galpy_pot = [
    ...     gpy.KeplerPotential(amp=u.Quantity(1e11, "Msun")),
    ...     gpy.KeplerPotential(amp=u.Quantity(8e11, "Msun"))
    ... ]
    >>> pot = gp.io.convert_potential(gp.io.GalaxLibrary, galpy_pot)
    >>> pot
    CompositePotential({'pot_0': KeplerPotential( ... ),
                        'pot_1': KeplerPotential( ... )})

    """  # noqa: E501
    return gpx.CompositePotential(
        {f"pot_{i}": galpy_to_galax(p) for i, p in enumerate(pot)}
    )


@dispatch  # type: ignore[misc]
def galax_to_galpy(pot: gpx.CompositePotential, /) -> list[gpy.Potential]:
    """Convert a :class:`~galax.potential.CompositePotential` to a list of :class:`~galpy.potential.Potential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> pot = gp.CompositePotential({
    ...     'pot_0': gp.KeplerPotential(m_tot=u.Quantity(1e11, "Msun"), units="galactic"),
    ...     'pot_1':  gp.KeplerPotential(m_tot=u.Quantity(8e11, "Msun"), units="galactic")
    ... })
    >>> galpy_pot = gp.io.convert_potential(gp.io.GalpyLibrary, pot)
    >>> galpy_pot
    [<galpy.potential...KeplerPotential object at ...>,
     <galpy.potential...KeplerPotential object at ...>]

    """  # noqa: E501
    return [galax_to_galpy(p) for p in pot.values()]


# ---------------------------
# Burkert potential


@dispatch  # type: ignore[misc]
def galpy_to_galax(pot: gpy.BurkertPotential, /) -> gpx.BurkertPotential:
    """Convert a :class:`~galpy.potential.BurkertPotential` to a :class:`~galax.potential.BurkertPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> galpy_pot = gpy.BurkertPotential(amp=u.Quantity(1e8, "Msun/kpc3"), a=1.0)
    >>> pot = gp.io.convert_potential(gp.io.GalaxLibrary, galpy_pot)
    >>> pot
    BurkertPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m=ConstantParameter( ... ),
      r_s=ConstantParameter( ... )
    )

    >>> pot.m(0)
    Quantity['mass'](Array(8.18153508e+10, dtype=float64), unit='solMass')

    >>> pot.r_s(0)
    Quantity['length'](Array(8., dtype=float64), unit='kpc')

    """  # noqa: E501
    # TODO: factor in the constants, e.g. G?
    return gpx.BurkertPotential.from_central_density(
        rho_0=Quantity(
            pot._amp * conversion.dens_in_msolpc3(pot._vo, pot._ro),  # noqa: SLF001
            "Msun / pc3",
        ),
        r_s=Quantity(pot.a * pot._ro, "kpc"),  # noqa: SLF001
        units="galactic",
    )


@dispatch  # type: ignore[misc]
def galax_to_galpy(pot: gpx.BurkertPotential, /) -> gpy.BurkertPotential:
    """Convert a :class:`~galax.potential.BurkertPotential` to a :class:`~galpy.potential.BurkertPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.BurkertPotential(
    ...     m=Quantity(1e8, "Msun"),
    ...     r_s=Quantity(1.0, "kpc"),
    ...     units="galactic"
    ... )
    >>> gp.io.convert_potential(gp.io.GalpyLibrary, pot)
    <galpy.potential...BurkertPotential object at ...>

    """  # noqa: E501
    pot = _error_if_not_all_constant_parameters(pot)

    return gpy.BurkertPotential(
        amp=convert(pot.rho0(0), AstropyQuantity),
        a=convert(pot.r_s(0), AstropyQuantity),
    )


# ---------------------------
# Hernquist potentials


@dispatch  # type: ignore[misc]
def galpy_to_galax(pot: gpy.HernquistPotential, /) -> gpx.HernquistPotential:
    """Convert a :class:`~galpy.potential.HernquistPotential` to a :class:`~galax.potential.HernquistPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> galpy_pot = gpy.HernquistPotential(amp=u.Quantity(1e11, "Msun"), a=1.0)
    >>> pot = gp.io.convert_potential(gp.io.GalaxLibrary, galpy_pot)
    >>> pot
    HernquistPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      r_s=ConstantParameter( ... )
    )

    """  # noqa: E501
    # TODO: factor in the constants, e.g. G?
    # TODO: unit management
    return gpx.HernquistPotential(
        m_tot=Quantity.from_(pot.mass(np.inf), "Msun"),
        r_s=Quantity.from_(pot.a, "kpc"),
        units="galactic",
    )


@dispatch  # type: ignore[misc]
def galax_to_galpy(pot: gpx.HernquistPotential, /) -> gpy.HernquistPotential:
    """Convert a :class:`~galax.potential.HernquistPotential` to a :class:`~galpy.potential.HernquistPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.HernquistPotential(
    ...     m_tot=Quantity(1e11, "Msun"),
    ...     r_s=Quantity(1.0, "kpc"),
    ...     units="galactic"
    ... )
    >>> gp.io.convert_potential(gp.io.GalpyLibrary, pot)
    <galpy.potential...HernquistPotential object at ...>

    """  # noqa: E501
    pot = _error_if_not_all_constant_parameters(pot)

    return gpy.HernquistPotential(
        amp=convert(pot.m_tot(0) * pot.constants["G"], AstropyQuantity),
        a=convert(pot.r_s(0), AstropyQuantity),
    )


# ---------------------------
# Isochrone potentials


@dispatch  # type: ignore[misc]
def galpy_to_galax(pot: gpy.IsochronePotential, /) -> gpx.IsochronePotential:
    """Convert a `galpy.potential.IsochronePotential` to a `galax.potential.IsochronePotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> galpy_pot = gpy.IsochronePotential(amp=u.Quantity(1e11, "Msun"), b=1.0)
    >>> pot = gp.io.convert_potential(gp.io.GalaxLibrary, galpy_pot)
    >>> pot
    IsochronePotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      b=ConstantParameter( ... )
    )

    >>> pot.m_tot(0)
    Quantity['mass'](Array(1.e+11, dtype=float64), unit='solMass')

    """  # noqa: E501
    return gpx.IsochronePotential(
        m_tot=_galpy_mass(pot),
        b=Quantity(pot.b * pot._ro, "kpc"),  # noqa: SLF001
        units="galactic",
    )


@dispatch  # type: ignore[misc]
def galax_to_galpy(pot: gpx.IsochronePotential, /) -> gpy.IsochronePotential:
    """Convert a `galax.potential.IsochronePotential` to a `galpy.potential.IsochronePotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.IsochronePotential(
    ...     m_tot=Quantity(1e11, "Msun"),
    ...     b=Quantity(1.0, "kpc"),
    ...     units="galactic"
    ... )
    >>> gp.io.convert_potential(gp.io.GalpyLibrary, pot)
    <galpy.potential...IsochronePotential object at ...>

    """  # noqa: E501
    pot = _error_if_not_all_constant_parameters(pot)

    return gpy.IsochronePotential(
        amp=convert(pot.m_tot(0) * pot.constants["G"], AstropyQuantity),
        b=convert(pot.b(0), AstropyQuantity),
    )


# ---------------------------
# Jaffe potentials


@dispatch  # type: ignore[misc]
def galpy_to_galax(pot: gpy.JaffePotential, /) -> gpx.JaffePotential:
    """Convert a `galpy.potential.JaffePotential` to a `galax.potential.JaffePotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> pot = gpy.JaffePotential(amp=u.Quantity(1e11, "Msun"), a=1.0)
    >>> gp.io.convert_potential(gp.io.GalaxLibrary, pot)
    JaffePotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m=ConstantParameter( ... ),
      r_s=ConstantParameter( ... )
    )

    """
    return gpx.JaffePotential(
        m=_galpy_mass(pot),
        r_s=Quantity(pot.a * pot._ro, "kpc"),  # noqa: SLF001
        units="galactic",
    )


@dispatch  # type: ignore[misc]
def galax_to_galpy(pot: gpx.JaffePotential, /) -> gpy.JaffePotential:
    """Convert a `galax.potential.JaffePotential` to a `galpy.potential.JaffePotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.JaffePotential(
    ...     m=Quantity(1e11, "Msun"),
    ...     r_s=Quantity(1.0, "kpc"),
    ...     units="galactic"
    ... )
    >>> gp.io.convert_potential(gp.io.GalpyLibrary, pot)
    <galpy.potential...JaffePotential object at ...>

    """
    pot = _error_if_not_all_constant_parameters(pot)

    return gpy.JaffePotential(
        amp=convert(pot.m(0) * pot.constants["G"], AstropyQuantity),
        a=convert(pot.r_s(0), AstropyQuantity),
    )


# ---------------------------
# Kepler potentials


@dispatch  # type: ignore[misc]
def galpy_to_galax(pot: gpy.KeplerPotential, /) -> gpx.KeplerPotential:
    """Convert a :class:`~galpy.potential.KeplerPotential` to a :class:`~galax.potential.KeplerPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> galpy_pot = gpy.KeplerPotential(amp=u.Quantity(1e11, "Msun"))
    >>> pot = gp.io.convert_potential(gp.io.GalaxLibrary, galpy_pot)
    >>> pot
    KeplerPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=...
    )
    >>> pot.m_tot(0)
    Quantity['mass'](Array(1.e+11, dtype=float64), unit='solMass')

    """  # noqa: E501
    # TODO: factor in the constants, e.g. G?
    # TODO: unit management
    return gpx.KeplerPotential(m_tot=pot.mass(np.inf), units="galactic")


@dispatch  # type: ignore[misc]
def galax_to_galpy(pot: gpx.KeplerPotential, /) -> gpy.KeplerPotential:
    """Convert a :class:`~galax.potential.KeplerPotential` to a :class:`~galpy.potential.KeplerPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=Quantity(1e11, "Msun"), units="galactic")
    >>> gp.io.convert_potential(gp.io.GalpyLibrary, pot)
    <galpy.potential...KeplerPotential object at ...>

    """  # noqa: E501
    pot = _error_if_not_all_constant_parameters(pot)

    # TODO: factor in the constants, e.g. G?
    return gpy.KeplerPotential(amp=pot.m_tot(0) * pot.constants["G"])


# ---------------------------
# Kuzmin potentials


@dispatch  # type: ignore[misc]
def galpy_to_galax(pot: gpy.KuzminDiskPotential, /) -> gpx.KuzminPotential:
    """Convert a :class:`~galpy.potential.KuzminDiskPotential` to a :class:`~galax.potentia.KuzminPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> galpy_pot = gpy.KuzminDiskPotential(amp=u.Quantity(1e11, "Msun"), a=1.0)
    >>> pot = gp.io.convert_potential(gp.io.GalaxLibrary, galpy_pot)
    >>> pot
    KuzminPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      a=ConstantParameter( ... )
    )

    """  # noqa: E501
    return gpx.KuzminPotential(
        m_tot=_galpy_mass(pot),
        a=Quantity.from_(pot._a * pot._ro, "kpc"),  # noqa: SLF001
        units="galactic",
    )


@dispatch  # type: ignore[misc]
def galax_to_galpy(pot: gpx.KuzminPotential, /) -> gpy.KuzminDiskPotential:
    """Convert a :class:`~galax.potentia.KuzminPotential` to a :class:`~galpy.potential.KuzminDiskPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.KuzminPotential(
    ...     m_tot=Quantity(1e11, "Msun"),
    ...     a=Quantity(1.0, "kpc"),
    ...     units="galactic"
    ... )
    >>> gp.io.convert_potential(gp.io.GalpyLibrary, pot)
    <galpy.potential...KuzminDiskPotential object at ...>

    """  # noqa: E501
    pot = _error_if_not_all_constant_parameters(pot)

    return gpy.KuzminDiskPotential(
        amp=convert(pot.m_tot(0) * pot.constants["G"], AstropyQuantity),
        a=convert(pot.a(0), AstropyQuantity),
    )


# ---------------------------
# Long & Murali potentials


@dispatch  # type: ignore[misc]
def galpy_to_galax(
    pot: gpy.SoftenedNeedleBarPotential, /
) -> gpx.LongMuraliBarPotential:
    """Convert a :class:`~galpy.potential.SoftenedNeedleBarPotential` to a :class:`~galax.potential.LongMuraliBarPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> galpy_pot = gpy.SoftenedNeedleBarPotential(amp=u.Quantity(1e11, "Msun"), a=1.0, b=0.5, c=0.25)
    >>> pot = gp.io.convert_potential(gp.io.GalaxLibrary, galpy_pot)
    >>> pot
    LongMuraliBarPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      a=ConstantParameter( ... ),
      b=ConstantParameter( ... ),
      c=ConstantParameter( ... ),
      alpha=ConstantParameter( ... )
    )

    """  # noqa: E501
    return gpx.LongMuraliBarPotential(
        m_tot=_galpy_mass(pot),
        a=Quantity.from_(pot._a * pot._ro, "kpc"),  # noqa: SLF001
        b=Quantity.from_(pot._b * pot._ro, "kpc"),  # noqa: SLF001
        c=Quantity.from_(np.sqrt(pot._c2) * pot._ro, "kpc"),  # noqa: SLF001
        units="galactic",
        alpha=Quantity.from_(pot._pa, "rad"),  # noqa: SLF001
    )


@dispatch  # type: ignore[misc]
def galax_to_galpy(
    pot: gpx.LongMuraliBarPotential, /
) -> gpy.SoftenedNeedleBarPotential:
    """Convert a :class:`~galax.potential.LongMuraliBarPotential` to a :class:`~galpy.potential.SoftenedNeedleBarPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.LongMuraliBarPotential(
    ...     m_tot=Quantity(1e11, "Msun"),
    ...     a=Quantity(1.0, "kpc"),
    ...     b=Quantity(0.5, "kpc"),
    ...     c=Quantity(0.25, "kpc"),
    ...     alpha=Quantity(0.0, "rad"),
    ...     units="galactic"
    ... )
    >>> gp.io.convert_potential(gp.io.GalpyLibrary, pot)
    <galpy.potential...SoftenedNeedleBarPotential object at ...>

    """  # noqa: E501
    pot = _error_if_not_all_constant_parameters(pot)

    return gpy.SoftenedNeedleBarPotential(
        amp=convert(pot.m_tot(0) * pot.constants["G"], AstropyQuantity),
        a=convert(pot.a(0), AstropyQuantity),
        b=convert(pot.b(0), AstropyQuantity),
        c=convert(pot.c(0), AstropyQuantity),
        pa=convert(pot.alpha(0), AstropyQuantity),
    )


# ---------------------------
# Miyamoto-Nagai potentials


@dispatch  # type: ignore[misc]
def galpy_to_galax(pot: gpy.MiyamotoNagaiPotential, /) -> gpx.MiyamotoNagaiPotential:
    """Convert a :class:`~galpy.potential.MiyamotoNagaiPotential` to a :class:`~galax.potential.MiyamotoNagaiPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> galpy_pot = gpy.MiyamotoNagaiPotential(amp=u.Quantity(1e11, "Msun"), a=1.0, b=0.5)
    >>> pot = gp.io.convert_potential(gp.io.GalaxLibrary, galpy_pot)
    >>> pot
    MiyamotoNagaiPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      a=ConstantParameter( ... ),
      b=ConstantParameter( ... )
    )

    """  # noqa: E501
    return gpx.MiyamotoNagaiPotential(
        m_tot=_galpy_mass(pot),
        a=Quantity.from_(pot._a * pot._ro, "kpc"),  # noqa: SLF001
        b=Quantity.from_(pot._b * pot._ro, "kpc"),  # noqa: SLF001
        units="galactic",
    )


@dispatch  # type: ignore[misc]
def galax_to_galpy(pot: gpx.MiyamotoNagaiPotential, /) -> gpy.MiyamotoNagaiPotential:
    """Convert a :class:`~galax.potential.MiyamotoNagaiPotential` to a :class:`~galpy.potential.MiyamotoNagaiPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.MiyamotoNagaiPotential(
    ...     m_tot=Quantity(1e11, "Msun"),
    ...     a=Quantity(1.0, "kpc"),
    ...     b=Quantity(0.5, "kpc"),
    ...     units="galactic"
    ... )
    >>> gp.io.convert_potential(gp.io.GalpyLibrary, pot)
    <galpy.potential...MiyamotoNagaiPotential object at ...>

    """  # noqa: E501
    pot = _error_if_not_all_constant_parameters(pot)

    return gpy.MiyamotoNagaiPotential(
        amp=convert(pot.m_tot(0) * pot.constants["G"], AstropyQuantity),
        a=convert(pot.a(0), AstropyQuantity),
        b=convert(pot.b(0), AstropyQuantity),
    )


# ---------------------------
# Null potentials


@dispatch  # type: ignore[misc]
def galpy_to_galax(_: gpy.NullPotential, /) -> gpx.NullPotential:
    """Convert a :class:`~galpy.potential.NullPotential` to a :class:`~galax.potential.NullPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import galax.potential as gp

    >>> galpy_pot = gpy.NullPotential()
    >>> pot = gp.io.convert_potential(gp.io.GalaxLibrary, galpy_pot)
    >>> pot
    NullPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...})
    )

    """  # noqa: E501
    return gpx.NullPotential()


@dispatch  # type: ignore[misc]
def galax_to_galpy(_: gpx.NullPotential, /) -> gpy.NullPotential:
    """Convert a :class:`~galax.potential.NullPotential` to a :class:`~galpy.potential.NullPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import galax.potential as gp

    >>> pot = gp.NullPotential()
    >>> gp.io.convert_potential(gp.io.GalpyLibrary, pot)
    <galpy.potential...NullPotential object at ...>

    """  # noqa: E501
    return gpy.NullPotential()


# ---------------------------
# Plummer potentials


@dispatch  # type: ignore[misc]
def galpy_to_galax(pot: gpy.PlummerPotential, /) -> gpx.PlummerPotential:
    """Convert a :class:`~galpy.potential.PlummerPotential` to a :class:`~galax.potential.PlummerPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> galpy_pot = gpy.PlummerPotential(amp=u.Quantity(1e11, "Msun"), b=1.0)
    >>> pot = gp.io.convert_potential(gp.io.GalaxLibrary, galpy_pot)
    >>> pot
    PlummerPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      b=ConstantParameter( ... )
    )

    """  # noqa: E501
    return gpx.PlummerPotential(
        m_tot=_galpy_mass(pot),
        b=Quantity.from_(pot._b * pot._ro, "kpc"),  # noqa: SLF001
        units="galactic",
    )


@dispatch  # type: ignore[misc]
def galax_to_galpy(pot: gpx.PlummerPotential, /) -> gpy.PlummerPotential:
    """Convert a :class:`~galax.potential.PlummerPotential` to a :class:`~galpy.potential.PlummerPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.PlummerPotential(
    ...     m_tot=Quantity(1e11, "Msun"),
    ...     b=Quantity(1.0, "kpc"),
    ...     units="galactic"
    ... )
    >>> gp.io.convert_potential(gp.io.GalpyLibrary, pot)
    <galpy.potential...PlummerPotential object at ...>

    """  # noqa: E501
    pot = _error_if_not_all_constant_parameters(pot)

    return gpy.PlummerPotential(
        amp=convert(pot.m_tot(0) * pot.constants["G"], AstropyQuantity),
        b=convert(pot.b(0), AstropyQuantity),
    )


# ---------------------------
# PowerLawCutoff potentials


@dispatch  # type: ignore[misc]
def galpy_to_galax(
    pot: gpy.PowerSphericalPotentialwCutoff, /
) -> gpx.PowerLawCutoffPotential:
    """Convert a :class:`~galpy.potential.PowerSphericalPotentialwCutoff` to a :class:`~galax.potential.PowerLawCutoffPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> galpy_pot = gpy.PowerSphericalPotentialwCutoff(amp=u.Quantity(1e6, "Msun / pc3"), alpha=1.0, rc=1.0)
    >>> pot = gp.io.convert_potential(gp.io.GalaxLibrary, galpy_pot)
    >>> pot
    PowerLawCutoffPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter( ... ),
      alpha=ConstantParameter( ... ),
      r_c=ConstantParameter( ... )
    )

    """  # noqa: E501
    return gpx.PowerLawCutoffPotential(
        m_tot=_galpy_mass(pot),
        alpha=Quantity(pot.alpha, ""),
        r_c=Quantity.from_(pot.rc * pot._ro, "kpc"),  # noqa: SLF001
        units="galactic",
    )


@dispatch  # type: ignore[misc]
def galax_to_galpy(
    pot: gpx.PowerLawCutoffPotential, /
) -> gpy.PowerSphericalPotentialwCutoff:
    """Convert a :class:`~galax.potential.PowerLawCutoffPotential` to a :class:`~galpy.potentia.PowerSphericalPotentialwCutoff`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.PowerLawCutoffPotential(
    ...     m_tot=Quantity(1e8, "Msun"),
    ...     alpha=Quantity(1.0, ""),
    ...     r_c=Quantity(1.0, "kpc"),
    ...     units="galactic"
    ... )
    >>> gp.io.convert_potential(gp.io.GalpyLibrary, pot)
    <galpy.potential...PowerSphericalPotentialwCutoff object at ...>

    """  # noqa: E501
    pot = _error_if_not_all_constant_parameters(pot)

    return gpy.PowerSphericalPotentialwCutoff(
        amp=convert(pot.density(Quantity([0.0, 0, 0], "kpc"), 0.0), AstropyQuantity),
        alpha=convert(pot.alpha(0), AstropyQuantity),
        rc=convert(pot.r_c(0), AstropyQuantity),
    )


# ---------------------------
# TODO: Logarithmic potentials


# -----------------------------------------------------------------------------
# NFW potentials


@dispatch  # type: ignore[misc]
def galpy_to_galax(pot: gpy.NFWPotential, /) -> gpx.NFWPotential:
    """Convert a :class:`~galpy.potential.NFWPotential` to a :class:`~galax.potential.NFWPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> galpy_pot = gpy.NFWPotential(amp=u.Quantity(1e11, "Msun"), a=1.0)
    >>> pot = gp.io.convert_potential(gp.io.GalaxLibrary, galpy_pot)
    >>> pot
    NFWPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      m=ConstantParameter( ... ),
      r_s=ConstantParameter( ... )
    )

    """  # noqa: E501
    return gpx.NFWPotential(
        m=_galpy_mass(pot),
        r_s=Quantity.from_(pot.a * pot._ro, "kpc"),  # noqa: SLF001
        units="galactic",
    )


@dispatch  # type: ignore[misc]
def galax_to_galpy(pot: gpx.NFWPotential, /) -> gpy.NFWPotential:
    """Convert a :class:`~galax.potential.NFWPotential` to a :class:`~galpy.potential.NFWPotential`.

    Examples
    --------
    >>> import galpy.potential as gpy
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.NFWPotential(
    ...     m=Quantity(1e11, "Msun"),
    ...     r_s=Quantity(1.0, "kpc"),
    ...     units="galactic"
    ... )
    >>> gp.io.convert_potential(gp.io.GalpyLibrary, pot)
    <galpy.potential...NFWPotential object at ...>

    """  # noqa: E501
    pot = _error_if_not_all_constant_parameters(pot)

    return gpy.NFWPotential(
        amp=convert(pot.m(0) * pot.constants["G"], AstropyQuantity),
        a=convert(pot.r_s(0), AstropyQuantity),
    )


# -----------------------------------------------------------------------------
# MW potentials

# ---------------------------
# TODO: Bovy MWPotential2014

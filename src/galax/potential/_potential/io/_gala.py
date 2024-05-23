"""Interoperability."""

__all__ = ["gala_to_galax"]

from functools import singledispatch
from typing import TypeVar

from packaging.version import Version

try:  # TODO: less hacky way of supporting optional dependencies
    import pytest
except ImportError:  # pragma: no cover
    pass
else:
    _ = pytest.importorskip("gala")

import gala.potential as gp
from gala.units import DimensionlessUnitSystem as GalaDimensionlessUnitSystem

import coordinax.operators as cxo
from coordinax.operators import IdentityOperator
from unxt import Quantity

import galax.potential as gpx
from galax.utils._optional_deps import HAS_GALA

##############################################################################
# GALA -> GALAX


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
    >>> gpx.io.gala_to_galax(gpot)
    CompositePotential({'disk': MiyamotoNagaiPotential( ... ),
                        'halo': NFWPotential( ... )})

    Hernquist potential:

    >>> gpot = gp.HernquistPotential(m=1e11 * u.Msun, c=10 * u.kpc, units=gu.galactic)
    >>> gpx.io.gala_to_galax(gpot)
    HernquistPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableDict({'G': ...}),
      m_tot=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ),
      r_s=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) ) )

    Isochrone potential:

    >>> gpot = gp.IsochronePotential(m=1e11 * u.Msun, b=10 * u.kpc, units=gu.galactic)
    >>> gpx.io.gala_to_galax(gpot)
    IsochronePotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableDict({'G': ...}),
      m_tot=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ),
      b=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) ) )

    Kepler potential:

    >>> gpot = gp.KeplerPotential(m=1e11 * u.Msun, units=gu.galactic)
    >>> gpx.io.gala_to_galax(gpot)
    KeplerPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableDict({'G': ...}),
      m_tot=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ) )

    >>> gpot = gp.LeeSutoTriaxialNFWPotential(
    ...     v_c=220, r_s=20, a=1, b=0.9, c=0.8, units=gu.galactic )
    >>> gpx.io.gala_to_galax(gpot)
    LeeSutoTriaxialNFWPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableDict({'G': ...}),
      m=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ),
      r_s=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) ),
      a1=ConstantParameter( unit=Unit(dimensionless), value=Quantity[...]( value=f64[], unit=Unit(dimensionless) ) ),
      a2=ConstantParameter( unit=Unit(dimensionless), value=Quantity[...]( value=f64[], unit=Unit(dimensionless) ) ),
      a3=ConstantParameter( unit=Unit(dimensionless), value=Quantity[...]( value=f64[], unit=Unit(dimensionless) ) )
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
      constants=ImmutableDict({'G': ...}),
      m_tot=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ),
      a=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) ),
      b=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) ) )

    NFW potential:

    >>> gpot = gp.NFWPotential(m=1e12, r_s=20, units=gu.galactic)
    >>> gpx.io.gala_to_galax(gpot)
    NFWPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableDict({'G': ...}),
      m=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ),
      r_s=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) ) )

    Null potential:

    >>> gpot = gp.NullPotential()
    >>> gpx.io.gala_to_galax(gpot)
    NullPotential( units=DimensionlessUnitSystem(),
                   constants=ImmutableDict({'G': ...}) )
    """  # noqa: E501
    msg = (
        "`gala_to_galax` does not have a registered function to convert "
        f"{pot.__class__.__name__!r} to a `galax.AbstractPotentialBase` instance."
    )
    raise NotImplementedError(msg)


# -----------------------
# Helper functions

PT = TypeVar("PT", bound=gpx.AbstractPotentialBase)


def _get_frame(pot: gp.PotentialBase, /) -> cxo.AbstractOperator:
    frame = cxo.GalileanSpatialTranslationOperator(
        Quantity(pot.origin, unit=pot.units["length"])
    )
    if pot.R is not None:
        frame = cxo.GalileanRotationOperator(pot.R) | frame
    return cxo.simplify_op(frame)


def _apply_frame(frame: cxo.AbstractOperator, pot: PT, /) -> PT | gpx.PotentialFrame:
    return (
        pot if isinstance(frame, IdentityOperator) else gpx.PotentialFrame(pot, frame)
    )


# -----------------------------------------------------------------------------
# General rules


@gala_to_galax.register
def _gala_to_galax_composite(pot: gp.CompositePotential, /) -> gpx.CompositePotential:
    """Convert a Gala CompositePotential to a Galax potential."""
    return gpx.CompositePotential(**{k: gala_to_galax(p) for k, p in pot.items()})


_GALA_TO_GALAX_REGISTRY: dict[type[gp.PotentialBase], type[gpx.AbstractPotential]] = {
    gp.IsochronePotential: gpx.IsochronePotential,
    gp.KeplerPotential: gpx.KeplerPotential,
    gp.KuzminPotential: gpx.KuzminPotential,
    gp.MiyamotoNagaiPotential: gpx.MiyamotoNagaiPotential,
    gp.PlummerPotential: gpx.PlummerPotential,
    gp.PowerLawCutoffPotential: gpx.PowerLawCutoffPotential,
}


@gala_to_galax.register(gp.IsochronePotential)
@gala_to_galax.register(gp.KeplerPotential)
@gala_to_galax.register(gp.KuzminPotential)
@gala_to_galax.register(gp.MiyamotoNagaiPotential)
@gala_to_galax.register(gp.PlummerPotential)
@gala_to_galax.register(gp.PowerLawCutoffPotential)
def _gala_to_galax_registered(
    gala: gp.PotentialBase, /
) -> gpx.AbstractPotential | gpx.PotentialFrame:
    """Convert a Gala potential to a Galax potential."""
    if isinstance(gala.units, GalaDimensionlessUnitSystem):
        msg = "Galax does not support converting dimensionless units."
        raise TypeError(msg)

    # TODO: this is a temporary solution. It would be better to map each
    # potential individually.
    params = dict(gala.parameters)
    params["m_tot"] = params.pop("m")

    pot = _GALA_TO_GALAX_REGISTRY[type(gala)](**params, units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


# -----------------------------------------------------------------------------
# Builtin potentials


@gala_to_galax.register
def _gala_to_galax_null(pot: gp.NullPotential, /) -> gpx.NullPotential:
    """Convert a Gala NullPotential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import galax.potential as gpx

    >>> gpot = gp.NullPotential()
    >>> gpx.io.gala_to_galax(gpot)
    NullPotential( units=DimensionlessUnitSystem(),
                   constants=ImmutableDict({'G': ...}) )

    """
    return gpx.NullPotential(units=pot.units)


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

        .. skip: start if(HAS_GALA < Version("1.8.2"), reason="Gala v1.8.2+")

        >>> gpot = gp.BurkertPotential(rho=4, r0=20, units=gu.galactic)
        >>> gpx.io.gala_to_galax(gpot)
        BurkertPotential(
        units=UnitSystem(kpc, Myr, solMass, rad),
        constants=ImmutableDict({'G': ...}),
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
    >>> gpx.io.gala_to_galax(gpot)
    HernquistPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableDict({'G': ...}),
      m_tot=ConstantParameter( ... ),
      r_s=ConstantParameter( ... )
    )
    """
    params = gala.parameters
    pot = gpx.HernquistPotential(m_tot=params["m"], r_s=params["c"], units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


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
    >>> gpx.io.gala_to_galax(gpot)
    JaffePotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableDict({'G': ...}),
      m=ConstantParameter( ... ),
      r_s=ConstantParameter( ... )
    )
    """
    params = gala.parameters
    pot = gpx.JaffePotential(m=params["m"], r_s=params["c"], units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


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
    >>> gpx.io.gala_to_galax(gpot)
    LongMuraliBarPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableDict({'G': Quantity...}),
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
    >>> gpx.io.gala_to_galax(gpot)
    SatohPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableDict({'G': ...}),
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
    >>> gpx.io.gala_to_galax(gpot)
    StoneOstriker15Potential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableDict({'G': ...}),
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
    >>> gpx.io.gala_to_galax(gpot)
    LogarithmicPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableDict({'G': ...}),
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
    >>> gpx.io.gala_to_galax(gpot)
    NFWPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableDict({'G': ...}),
      m=ConstantParameter( unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass")) ),
      r_s=ConstantParameter( unit=Unit("kpc"), value=Quantity[...](value=f64[], unit=Unit("kpc")) )
    )

    """  # noqa: E501
    params = gala.parameters
    pot = gpx.NFWPotential(m=params["m"], r_s=params["r_s"], units=gala.units)
    return _apply_frame(_get_frame(gala), pot)


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
    >>> gpx.io.gala_to_galax(gpot)
    LeeSutoTriaxialNFWPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableDict({'G': ...}),
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


# -----------------------------------------------------------------------------
# MW potentials


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
    >>> gpx.io.gala_to_galax(gpot)
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


@gala_to_galax.register
def _gala_to_galax_lm10(pot: gp.LM10Potential, /) -> gpx.LM10Potential:
    """Convert a Gala LM10Potential to a Galax potential.

    Examples
    --------
    >>> import gala.potential as gp
    >>> import galax.potential as gpx

    >>> gpot = gp.LM10Potential()
    >>> gpx.io.gala_to_galax(gpot)
    LM10Potential({'disk': MiyamotoNagaiPotential( ... ),
                   'bulge': HernquistPotential( ... ),
                   'halo': LMJ09LogarithmicPotential( ... )})

    """
    return gpx.LM10Potential(
        disk=gala_to_galax(pot["disk"]),
        bulge=gala_to_galax(pot["bulge"]),
        halo=gala_to_galax(pot["halo"]),
    )


@gala_to_galax.register
def _gala_to_galax_mw(pot: gp.MilkyWayPotential, /) -> gpx.MilkyWayPotential:
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
    return gpx.MilkyWayPotential(
        disk=gala_to_galax(pot["disk"]),
        halo=gala_to_galax(pot["halo"]),
        bulge=gala_to_galax(pot["bulge"]),
        nucleus=gala_to_galax(pot["nucleus"]),
    )

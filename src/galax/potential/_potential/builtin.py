"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "BarPotential",
    "HernquistPotential",
    "IsochronePotential",
    "KeplerPotential",
    "LeeSutoTriaxialNFWPotential",
    "MiyamotoNagaiPotential",
    "NFWPotential",
    "NullPotential",
    "TriaxialHernquistPotential",
]

from dataclasses import KW_ONLY
from functools import partial
from typing import final

import equinox as eqx
import jax
from quax import quaxify

import quaxed.array_api as xp
import quaxed.lax as qlax
from unxt import AbstractUnitSystem, Quantity, unitsystem
from unxt.unitsystems import galactic

import galax.typing as gt
from galax.potential._potential.base import default_constants
from galax.potential._potential.core import AbstractPotential
from galax.potential._potential.param import AbstractParameter, ParameterField
from galax.utils import ImmutableDict
from galax.utils._jax import vectorize_method

# -------------------------------------------------------------------


@final
class BarPotential(AbstractPotential):
    """Rotating bar potentil, with hard-coded rotation.

    Eq 8a in https://articles.adsabs.harvard.edu/pdf/1992ApJ...397...44L
    Rz according to https://en.wikipedia.org/wiki/Rotation_matrix
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    Omega: AbstractParameter = ParameterField(dimensions="frequency")  # type: ignore[assignment]
    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    # TODO: inputs w/ units
    @quaxify  # type: ignore[misc]
    @partial(jax.jit)
    @vectorize_method(signature="(3),()->()")
    def _potential_energy(self, q: gt.QVec3, t: gt.RealQScalar, /) -> gt.FloatQScalar:
        ## First take the simulation frame coordinates and rotate them by Omega*t
        ang = -self.Omega(t) * t
        rotation_matrix = xp.asarray(
            [
                [xp.cos(ang), -xp.sin(ang), 0],
                [xp.sin(ang), xp.cos(ang), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        q_corot = xp.matmul(rotation_matrix, q)

        a = self.a(t)
        b = self.b(t)
        c = self.c(t)
        T_plus = xp.sqrt(
            (a + q_corot[0]) ** 2
            + q_corot[1] ** 2
            + (b + xp.sqrt(c**2 + q_corot[2] ** 2)) ** 2
        )
        T_minus = xp.sqrt(
            (a - q_corot[0]) ** 2
            + q_corot[1] ** 2
            + (b + xp.sqrt(c**2 + q_corot[2] ** 2)) ** 2
        )

        # potential in a corotating frame
        return (self.constants["G"] * self.m(t) / (2.0 * a)) * xp.log(
            (q_corot[0] - a + T_minus) / (q_corot[0] + a + T_plus),
        )


# -------------------------------------------------------------------


@final
class HernquistPotential(AbstractPotential):
    """Hernquist Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    @partial(jax.jit)
    def _potential_energy(  # TODO: inputs w/ units
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.BatchFloatQScalar:
        r = xp.linalg.vector_norm(q, axis=-1)
        return -self.constants["G"] * self.m(t) / (r + self.c(t))


# -------------------------------------------------------------------


@final
class IsochronePotential(AbstractPotential):
    """Isochrone Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    @partial(jax.jit)
    def _potential_energy(  # TODO: inputs w/ units
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.BatchFloatQScalar:
        r = xp.linalg.vector_norm(q, axis=-1)
        b = self.b(t)
        return -self.constants["G"] * self.m(t) / (b + xp.sqrt(r**2 + b**2))


# -------------------------------------------------------------------


@final
class KeplerPotential(AbstractPotential):
    r"""The Kepler potential for a point mass.

    .. math::
        \Phi = -\frac{G M(t)}{r}
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    @partial(jax.jit)
    def _potential_energy(  # TODO: inputs w/ units
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.BatchFloatQScalar:
        r = xp.linalg.vector_norm(q, axis=-1)
        return -self.constants["G"] * self.m(t) / r


# -------------------------------------------------------------------


@final
class MiyamotoNagaiPotential(AbstractPotential):
    """Miyamoto-Nagai Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    @partial(jax.jit)
    def _potential_energy(
        self: "MiyamotoNagaiPotential", q: gt.QVec3, t: gt.RealQScalar, /
    ) -> gt.FloatQScalar:
        R2 = q[..., 0] ** 2 + q[..., 1] ** 2
        zp2 = (xp.sqrt(q[..., 2] ** 2 + self.b(t) ** 2) + self.a(t)) ** 2
        return -self.constants["G"] * self.m(t) / xp.sqrt(R2 + zp2)


# -------------------------------------------------------------------


@final
class NFWPotential(AbstractPotential):
    """NFW Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    @partial(jax.jit)
    def _potential_energy(  # TODO: inputs w/ units
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.BatchFloatQScalar:
        v_h2 = -self.constants["G"] * self.m(t) / self.r_s(t)
        r = xp.linalg.vector_norm(q, axis=-1)
        m = r / self.r_s(t)
        return v_h2 * xp.log(1.0 + m) / m


_log2 = xp.log(xp.asarray(2.0))


@final
class LeeSutoTriaxialNFWPotential(AbstractPotential):
    """Approximate triaxial (in the density) NFW potential.

    Approximation of a Triaxial NFW Potential with the flattening in the
    density, not the potential. See Lee & Suto (2003) for details.

    .. warning::

        This potential is only physical for `a1 >= a2 >= a3`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.LeeSutoTriaxialNFWPotential(
    ...    m=Quantity(1e11, "Msun"), r_s=Quantity(15, "kpc"),
    ...    a1=1, a2=0.9, a3=0.8, units="galactic")

    >>> q = Quantity([1, 0, 0], "kpc")
    >>> t = Quantity(0, "Gyr")
    >>> pot.potential_energy(q, t).decompose(pot.units)
    Quantity['specific energy'](Array(-0.14620419, dtype=float64), unit='kpc2 / Myr2')

    >>> q = Quantity([0, 1, 0], "kpc")
    >>> pot.potential_energy(q, t).decompose(pot.units)
    Quantity['specific energy'](Array(-0.14593972, dtype=float64), unit='kpc2 / Myr2')

    >>> q = Quantity([0, 0, 1], "kpc")
    >>> pot.potential_energy(q, t).decompose(pot.units)
    Quantity['specific energy'](Array(-0.14570309, dtype=float64), unit='kpc2 / Myr2')
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    r"""Scall mass.

    This is the mass corresponding to the circular velocity at the scale radius.
    :math:`v_c = \sqrt{G M / r_s}`
    """

    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale radius."""

    a1: AbstractParameter = ParameterField(
        dimensions="dimensionless",
        default=Quantity(1.0, ""),  # type: ignore[assignment]
    )
    """Major axis."""

    a2: AbstractParameter = ParameterField(
        dimensions="dimensionless",
        default=Quantity(1.0, ""),  # type: ignore[assignment]
    )
    """Intermediate axis."""

    a3: AbstractParameter = ParameterField(
        dimensions="dimensionless",
        default=Quantity(1.0, ""),  # type: ignore[assignment]
    )
    """Minor axis."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    def __check_init__(self) -> None:
        t = Quantity(0.0, "Myr")
        _ = eqx.error_if(
            t,
            (self.a1(t) < self.a2(t)) or (self.a2(t) < self.a3(t)),
            "a1 >= a2 >= a3 is required",
        )

    @partial(jax.jit)
    def _potential_energy(  # TODO: inputs w/ units
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.BatchFloatQScalar:
        # https://github.com/adrn/gala/blob/2067009de41518a71c674d0252bc74a7b2d78a36/gala/potential/potential/builtin/builtin_potentials.c#L1472
        # Evaluate the parameters
        r_s = self.r_s(t)
        v_c2 = self.constants["G"] * self.m(t) / r_s
        a1, a2, a3 = self.a1(t), self.a2(t), self.a3(t)

        # 1- eccentricities
        e_b2 = 1 - xp.square(a2 / a1)
        e_c2 = 1 - xp.square(a3 / a1)

        # The potential at the origin
        phi0 = v_c2 / (_log2 - 0.5 + (_log2 - 0.75) * (e_b2 + e_c2))

        # The potential at the given position
        r = xp.linalg.vector_norm(q, axis=-1)
        u = r / r_s

        # The functions F1, F2, and F3 and some useful quantities
        log1pu = xp.log(1 + u)
        u2 = u**2
        um3 = u ** (-3)
        costh2 = q[..., 2] ** 2 / r**2  # z^2 / r^2
        sinthsinphi2 = q[..., 1] ** 2 / r**2  # (sin(theta) * sin(phi))^2
        # Note that ꜛ is safer than computing the separate pieces, as it avoids
        # x=y=0, z!=0, which would result in a NaN.

        F1 = -log1pu / u
        F2 = -1.0 / 3 + (2 * u2 - 3 * u + 6) / (6 * u2) + (1 / u - um3) * log1pu
        F3 = (u2 - 3 * u - 6) / (2 * u2 * (1 + u)) + 3 * um3 * log1pu

        # Select the output, r=0 is a special case.
        out: gt.BatchFloatQScalar = phi0 * qlax.select(
            u == 0,
            xp.ones_like(u),
            (
                F1
                + (e_b2 + e_c2) / 2 * F2
                + (e_b2 * sinthsinphi2 + e_c2 * costh2) / 2 * F3
            ),
        )
        return out


# -------------------------------------------------------------------


@final
class NullPotential(AbstractPotential):
    """Null potential, i.e. no potential."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    @partial(jax.jit)
    def _potential_energy(  # TODO: inputs w/ units
        self,
        q: gt.BatchQVec3,
        t: gt.BatchableRealQScalar,  # noqa: ARG002
        /,
    ) -> gt.BatchFloatQScalar:
        return Quantity(  # TODO: better unit handling
            xp.zeros(q.shape[:-1], dtype=q.dtype), galactic["specific energy"]
        )


# -------------------------------------------------------------------


@final
class TriaxialHernquistPotential(AbstractPotential):
    """Triaxial Hernquist Potential.

    Parameters
    ----------
    m : :class:`~galax.potential.AbstractParameter`['mass']
        Mass parameter. This can be a
        :class:`~galax.potential.AbstractParameter` or an appropriate callable
        or constant, like a Quantity. See
        :class:`~galax.potential.ParameterField` for details.
    c : :class:`~galax.potential.AbstractParameter`['length']
        A scale length that determines the concentration of the system.  This
        can be a :class:`~galax.potential.AbstractParameter` or an appropriate
        callable or constant, like a Quantity. See
        :class:`~galax.potential.ParameterField` for details.
    q1 : :class:`~galax.potential.AbstractParameter`['length']
        Scale length in the y direction. This can be a
        :class:`~galax.potential.AbstractParameter` or an appropriate callable
        or constant, like a Quantity. See
        :class:`~galax.potential.ParameterField` for details.
    a2 : :class:`~galax.potential.AbstractParameter`['length']
        Scale length in the z direction. This can be a
        :class:`~galax.potential.AbstractParameter` or an appropriate callable
        or constant, like a Quantity. See
        :class:`~galax.potential.ParameterField` for details.

    units : :class:`~unxt.AbstractUnitSystem`, keyword-only
        The unit system to use for the potential.  This parameter accepts a
        :class:`~unxt.AbstractUnitSystem` or anything that can be converted to a
        :class:`~unxt.AbstractUnitSystem` using :func:`~unxt.unitsystem`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from galax.potential import TriaxialHernquistPotential

    >>> pot = TriaxialHernquistPotential(m=Quantity(1e12, "Msun"), c=Quantity(8, "kpc"),
    ...                                  q1=1, q2=0.5, units="galactic")

    >>> q = Quantity([1, 0, 0], "kpc")
    >>> t = Quantity(0, "Gyr")
    >>> pot.potential_energy(q, t)
    Quantity['specific energy'](Array(-0.49983357, dtype=float64), unit='kpc2 / Myr2')
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Mass of the potential."""

    c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale a scale length that determines the concentration of the system."""

    q1: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=Quantity(1, ""),
        dimensions="dimensionless",
    )
    """Scale length in the y direction divided by ``c``."""

    q2: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=Quantity(1, ""),
        dimensions="dimensionless",
    )
    """Scale length in the z direction divided by ``c``."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    """The unit system to use for the potential."""

    constants: ImmutableDict[Quantity] = eqx.field(
        converter=ImmutableDict, default=default_constants
    )
    """The constants used by the potential."""

    @partial(jax.jit)
    def _potential_energy(  # TODO: inputs w/ units
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.BatchFloatQScalar:
        c, q1, q2 = self.c(t), self.q1(t), self.q2(t)
        c = eqx.error_if(c, c.value <= 0, "c must be positive")

        rprime = xp.sqrt(q[..., 0] ** 2 + (q[..., 1] / q1) ** 2 + (q[..., 2] / q2) ** 2)
        return -self.constants["G"] * self.m(t) / (rprime + c)

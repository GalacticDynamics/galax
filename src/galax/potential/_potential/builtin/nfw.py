"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "NFWPotential",
    "LeeSutoTriaxialNFWPotential",
    "Vogelsberger08TriaxialNFWPotential",
]

from dataclasses import KW_ONLY
from functools import partial
from typing import final

import equinox as eqx
import jax

import quaxed.array_api as xp
import quaxed.lax as qlax
from unxt import AbstractUnitSystem, Quantity, unitsystem

import galax.typing as gt
from galax.potential._potential.base import default_constants
from galax.potential._potential.core import AbstractPotential
from galax.potential._potential.param import AbstractParameter, ParameterField
from galax.utils import ImmutableDict

_log2 = xp.log(xp.asarray(2.0))

# -------------------------------------------------------------------


@final
class NFWPotential(AbstractPotential):
    r"""NFW Potential.

    .. math::

        \rho(r) = -\frac{G M}{r_s} \frac{r_s}{r} \log(1 + \frac{r}{r_s})
        \Phi(r) = -\frac{G M}{r_s} \frac{r_s}{r} \log(1 + \frac{r}{r_s})

    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Mass parameter. This is NOT the total mass."""

    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale radius of the potential."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    @partial(jax.jit)
    def _potential_energy(  # TODO: inputs w/ units
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.BatchFloatQScalar:
        r"""Potential energy.

        .. math::

            \Phi(r) = -\frac{G M}{r_s} \frac{r_s}{r} \log(1 + \frac{r}{r_s})
        """
        r = xp.linalg.vector_norm(q, axis=-1)
        r_s = self.r_s(t)
        u = r / r_s
        v_h2 = self.constants["G"] * self.m(t) / r_s
        return -v_h2 * xp.log(1.0 + u) / u

    @partial(jax.jit)
    def _density(
        self, q: gt.BatchQVec3, /, t: gt.BatchRealQScalar | gt.RealQScalar
    ) -> gt.BatchFloatQScalar:
        r"""Density.

        .. math::

            v_{h2} = -\frac{G M}{r_s}
            \rho_0 = \frac{v_{h2}}{4 \pi G r_s^2}
            \rho(r) = \frac{\rho_0}{u (1 + u)^2}
        """
        r = xp.linalg.vector_norm(q, axis=-1)
        r_s = self.r_s(t)
        rho0 = self.m(t) / (4 * xp.pi * r_s**3)
        u = r / r_s
        return rho0 / u / (1 + u) ** 2


# -------------------------------------------------------------------


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


class Vogelsberger08TriaxialNFWPotential(AbstractPotential):
    """Triaxial NFW Potential from DOI 10.1111/j.1365-2966.2007.12746.x."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    r"""Scale mass."""
    # TODO: note the different definitions of m.

    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale radius."""

    q1: AbstractParameter = ParameterField(
        dimensions="dimensionless",
        default=Quantity(1.0, ""),  # type: ignore[assignment]
    )
    """y/x axis ratio.

    The z/x axis ratio is defined as :math:`q_2^2 = 3 - q_1^2`
    """

    a_r: AbstractParameter = ParameterField(
        dimensions="dimensionless",
        default=Quantity(1.0, ""),  # type: ignore[assignment]
    )
    """Transition radius relative to :math:`r_s`.

    :math:`r_a = a_r r_s  is a transition scale where the potential shape
    changes from ellipsoidal to near spherical.
    """

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    @partial(jax.jit, inline=True)
    def _r_e(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar
    ) -> gt.BatchFloatQScalar:
        q1sq = self.q1(t) ** 2
        q2sq = 3 - q1sq
        return xp.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2 / q1sq + q[..., 2] ** 2 / q2sq)

    @partial(jax.jit, inline=True)
    def _r_tilde(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar
    ) -> gt.BatchFloatQScalar:
        r_a = self.a_r(t) * self.r_s(t)
        r_e = self._r_e(q, t)
        r = xp.linalg.vector_norm(q, axis=-1)
        return (r_a + r) * r_e / (r_a + r_e)

    @partial(jax.jit)
    def _potential_energy(
        self: "Vogelsberger08TriaxialNFWPotential", q: gt.QVec3, t: gt.RealQScalar, /
    ) -> gt.FloatQScalar:
        r = self._r_tilde(q, t)
        return -self.constants["G"] * self.m(t) * xp.log(1.0 + r / self.r_s(t)) / r

"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "BurkertPotential",
    "HernquistPotential",
    "IsochronePotential",
    "JaffePotential",
    "KeplerPotential",
    "PlummerPotential",
    "PowerLawCutoffPotential",
    "StoneOstriker15Potential",
    "TriaxialHernquistPotential",
]

from dataclasses import KW_ONLY
from functools import partial
from typing import Annotated as Ann, Any, final
from typing_extensions import Doc

import equinox as eqx
import jax

import quaxed.lax as qlax
import quaxed.numpy as jnp
import quaxed.scipy.special as qsp
import unxt as u
from unxt.unitsystems import AbstractUnitSystem
from xmmutablemap import ImmutableMap

import galax.typing as gt
from .const import _burkert_const
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.core import AbstractParameter
from galax.potential._src.params.field import ParameterField

# -------------------------------------------------------------------


@final
class BurkertPotential(AbstractSinglePotential):
    """Burkert Potential.

    https://ui.adsabs.harvard.edu/abs/1995ApJ...447L..25B/abstract,
    https://iopscience.iop.org/article/10.1086/309140/fulltext/50172.text.html.
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    r"""Characteristic mass of the potential.

    .. math::

        m0 = \pi \rho_0 r_s^3 (3 \log(2) - \pi / 2)

    """

    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale radius"""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(
        self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
    ) -> gt.SpecificEnergyBtSz0:
        m, r_s = self.m(t), self.r_s(t)
        x = jnp.linalg.vector_norm(q, axis=-1) / r_s
        xinv = 1 / x
        return -(self.constants["G"] * m / (r_s * _burkert_const)) * (
            jnp.pi
            - 2 * (1 + xinv) * jnp.atan(x).value
            + 2 * (1 + xinv) * jnp.log(1 + x)
            - (1 - xinv) * jnp.log(1 + x**2)
        )

    @partial(jax.jit)
    def _density(
        self, q: gt.BtQuSz3, t: gt.BtRealQuSz0 | gt.RealQuSz0, /
    ) -> gt.BtFloatQuSz0:
        m, r_s = self.m(t), self.r_s(t)
        r = jnp.linalg.vector_norm(q, axis=-1)
        return m / (jnp.pi * _burkert_const) / ((r + r_s) * (r**2 + r_s**2))

    @partial(jax.jit)
    def _mass(
        self, q: gt.BtQuSz3, /, t: gt.BtRealQuSz0 | gt.RealQuSz0
    ) -> gt.BtFloatQuSz0:
        x = jnp.linalg.vector_norm(q, axis=-1) / self.r_s(t)
        return (
            self.m(t)
            / _burkert_const
            * (-2 * jnp.atan(x) + 2 * jnp.log(1 + x) + jnp.log(1 + x**2))
        )

    # -------------------------------------------------------------------

    def rho0(self, t: gt.BtRealQuSz0 | gt.RealQuSz0) -> gt.BtFloatQuSz0:
        r"""Central density of the potential.

        .. math::

            m0 = \pi \rho_0 r_s^3 (3 \log(2) - \pi / 2)
        """
        return self.m(t) / (jnp.pi * self.r_s(t) ** 3 * _burkert_const)

    # -------------------------------------------------------------------
    # Constructors

    @classmethod
    def from_central_density(
        cls, rho_0: u.Quantity, r_s: u.Quantity, **kwargs: Any
    ) -> "BurkertPotential":
        r"""Create a Burkert potential from the central density.

        Parameters
        ----------
        rho_0 : :class:`~unxt.Quantity`[mass density]
            Central density.
        r_s : :class:`~unxt.Quantity`[length]
            Scale radius.

        Returns
        -------
        :class:`~galax.potential.BurkertPotential`
            Burkert potential.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> rho_0 = u.Quantity(1e6, "Msun / kpc3")
        >>> r_s = u.Quantity(1, "kpc")
        >>> pot = gp.BurkertPotential.from_central_density(rho_0, r_s, units="galactic")
        >>> pot
        BurkertPotential(
            units=LTMAUnitSystem( length=Unit("kpc"), ...),
            constants=ImmutableMap({'G': ...}),
            m=ConstantParameter( ... ),
            r_s=ConstantParameter( ... )
        )

        """
        m = jnp.pi * rho_0 * r_s**3 * _burkert_const
        return cls(m=m, r_s=r_s, **kwargs)


# -------------------------------------------------------------------


@final
class HernquistPotential(AbstractSinglePotential):
    """Hernquist Potential."""

    m_tot: Ann[AbstractParameter, Doc("Total mass of the potential.")] = ParameterField(  # type: ignore[assignment]
        dimensions="mass"
    )

    r_s: Ann[AbstractParameter, Doc("Scale radius")] = ParameterField(
        dimensions="length"
    )  # type: ignore[assignment]

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(
        self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
    ) -> gt.SpecificEnergyBtSz0:
        r = jnp.linalg.vector_norm(q, axis=-1)
        return -self.constants["G"] * self.m_tot(t) / (r + self.r_s(t))

    @partial(jax.jit)
    def _density(self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /) -> gt.BtFloatQuSz0:
        r_s = self.r_s(t)
        x = jnp.linalg.vector_norm(q, axis=-1) / r_s
        rho0 = self.m_tot(t) / (2 * jnp.pi * r_s**3)
        return rho0 / (x * (1 + x) ** 3)


# -------------------------------------------------------------------


@final
class IsochronePotential(AbstractSinglePotential):
    r"""Isochrone Potential.

    .. math::

        \Phi = -\frac{G M(t)}{r_s + \sqrt{r^2 + r_s^2}}
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Total mass of the potential."""

    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    r"""Scale radius of the potential.

    The value of :math:`r_s` defines the transition between the inner, more
    harmonic oscillator-like behavior of the potential, and the outer, :math:`1
    / r` Keplerian falloff.
    """

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
    ) -> gt.SpecificEnergyBtSz0:
        m, r_s = self.m_tot(t), self.r_s(t)
        r = jnp.linalg.vector_norm(q, axis=-1)
        return -self.constants["G"] * m / (r_s + jnp.sqrt(r**2 + r_s**2))


# -------------------------------------------------------------------


@final
class JaffePotential(AbstractSinglePotential):
    """Jaffe Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    @partial(jax.jit)
    def _potential(
        self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
    ) -> gt.SpecificEnergyBtSz0:
        r_s = self.r_s(t)
        r = jnp.linalg.vector_norm(q, axis=-1)
        return -self.constants["G"] * self.m(t) / r_s * jnp.log(1 + r_s / r)


# -------------------------------------------------------------------


@final
class KeplerPotential(AbstractSinglePotential):
    r"""The Kepler potential for a point mass.

    .. math::

        \Phi = -\frac{G M(t)}{r}
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Total mass of the potential."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
    ) -> gt.SpecificEnergyBtSz0:
        r = jnp.linalg.vector_norm(q, axis=-1)
        return -self.constants["G"] * self.m_tot(t) / r

    @partial(jax.jit)
    def _density(
        self, q: gt.BtQuSz3, t: gt.BtRealQuSz0 | gt.RealQuSz0, /
    ) -> gt.BtFloatQuSz0:
        r = jnp.linalg.vector_norm(q, axis=-1)
        m = self.m_tot(t)
        pred = jnp.logical_or(  # are we at the origin with non-zero mass?
            jnp.greater(r, jnp.zeros_like(r)), jnp.equal(m, jnp.zeros_like(m))
        )
        return u.Quantity(
            qlax.select(
                pred,
                jnp.zeros_like(r.value),
                jnp.full_like(r.value, fill_value=jnp.inf),
            ),
            self.units["mass density"],
        )


# -------------------------------------------------------------------


@final
class PlummerPotential(AbstractSinglePotential):
    """Plummer Potential."""

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(
        self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
    ) -> gt.SpecificEnergyBtSz0:
        r2 = jnp.linalg.vector_norm(q, axis=-1) ** 2
        return -self.constants["G"] * self.m_tot(t) / jnp.sqrt(r2 + self.r_s(t) ** 2)


# -------------------------------------------------------------------


@partial(jax.jit)
def _safe_gamma_inc(a: u.Quantity, x: u.Quantity) -> u.Quantity:  # TODO: types
    return qsp.gammainc(a, x) * qsp.gamma(a)


@final
class PowerLawCutoffPotential(AbstractSinglePotential):
    r"""A spherical power-law density profile with an exponential cutoff.

    .. math::

        \rho(r) = \frac{G M}{2\pi \Gamma((3-\alpha)/2) r_c^3} \left(\frac{r_c}{r}\right)^\alpha \exp{-(r / r_c)^2}

    Parameters
    ----------
    m_tot : :class:`~unxt.Quantity`[mass]
        Total mass.
    alpha : :class:`~unxt.Quantity`[dimensionless]
        Power law index. Must satisfy: ``0 <= alpha < 3``.
    r_c : :class:`~unxt.Quantity`[length]
        Cutoff radius.
    """  # noqa: E501

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Total mass of the potential."""

    alpha: AbstractParameter = ParameterField(dimensions="dimensionless")  # type: ignore[assignment]
    """Power law index. Must satisfy: ``0 <= alpha < 3``"""

    r_c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Cutoff radius."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(
        self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
    ) -> gt.SpecificEnergyBtSz0:
        m, a, r_c = self.m_tot(t), 0.5 * self.alpha(t), self.r_c(t)
        r = jnp.linalg.vector_norm(q, axis=-1)
        rp2 = (r / r_c) ** 2

        return (self.constants["G"] * m) * (
            (a - 1.5) * _safe_gamma_inc(1.5 - a, rp2) / (r * qsp.gamma(2.5 - a))
            + _safe_gamma_inc(1 - a, rp2) / (r_c * qsp.gamma(1.5 - a))
        )


# -------------------------------------------------------------------


class StoneOstriker15Potential(AbstractSinglePotential):
    r"""StoneOstriker15Potential(m, r_c, r_h, units=None, origin=None, R=None).

    Stone potential from `Stone & Ostriker (2015)
    <http://dx.doi.org/10.1088/2041-8205/806/2/L28>`_.

    .. math::

        \Phi = -\frac{2 G m}{\pi (r_h - r_c)} \left(
            \frac{r_h}{r} \tan^{-1}(\frac{r}{r_h})
            - \frac{r_c}{r} \tan^{-1}(\frac{r}{r_c})
            + \frac{1}{2} \log(\frac{r^2 + r_h^2}{r^2 + r_c^2})
            \right)

    """

    #: Total mass.
    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]

    #: Core radius.
    r_c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    #: Halo radius.
    r_h: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    # def __check_init__(self) -> None:
    #     _ = eqx.error_if(self.r_c, self.r_c.value >= self.r_h.value, "Core radius must be less than halo radius")   # noqa: E501, ERA001

    @partial(jax.jit)
    def _potential(
        self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
    ) -> gt.SpecificEnergyBtSz0:
        r_h = self.r_h(t)
        r_c = self.r_c(t)
        r = jnp.linalg.vector_norm(q, axis=-1)
        A = -2 * self.constants["G"] * self.m_tot(t) / (jnp.pi * (r_h - r_c))
        return A * (
            (r_h / r) * u.ustrip("rad", jnp.atan2(r, r_h))
            - (r_c / r) * u.ustrip("rad", jnp.atan2(r, r_c))
            + 0.5 * jnp.log((r**2 + r_h**2) / (r**2 + r_c**2))
        )


# -------------------------------------------------------------------


@final
class TriaxialHernquistPotential(AbstractSinglePotential):
    """Triaxial Hernquist Potential.

    Parameters
    ----------
    m_tot : :class:`~galax.potential.AbstractParameter`['mass']
        Mass parameter. This can be a
        :class:`~galax.potential.AbstractParameter` or an appropriate callable
        or constant, like a Quantity. See
        :class:`~galax.potential.ParameterField` for details.
    r_s : :class:`~galax.potential.AbstractParameter`['length']
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
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.TriaxialHernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                                     r_s=u.Quantity(8, "kpc"), q1=1, q2=0.5,
    ...                                     units="galactic")

    >>> q = u.Quantity([1, 0, 0], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.potential(q, t)
    Quantity[...](Array(-0.49983357, dtype=float64), unit='kpc2 / Myr2')
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Mass of the potential."""

    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale a scale length that determines the concentration of the system."""

    # TODO: move to a triaxial wrapper
    q1: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(1.0, ""), dimensions="dimensionless"
    )
    """Scale length in the y direction divided by ``c``."""

    q2: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(1.0, ""), dimensions="dimensionless"
    )
    """Scale length in the z direction divided by ``c``."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        converter=ImmutableMap, default=default_constants
    )

    @partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
    ) -> gt.SpecificEnergyBtSz0:
        r_s, q1, q2 = self.r_s(t), self.q1(t), self.q2(t)
        r_s = eqx.error_if(r_s, r_s.value <= 0, "r_s must be positive")

        rprime = jnp.sqrt(
            q[..., 0] ** 2 + (q[..., 1] / q1) ** 2 + (q[..., 2] / q2) ** 2
        )
        return -self.constants["G"] * self.m_tot(t) / (rprime + r_s)

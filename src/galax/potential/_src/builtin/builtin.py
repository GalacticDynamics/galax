"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "BurkertPotential",
    "HarmonicOscillatorPotential",
    "HenonHeilesPotential",
    "HernquistPotential",
    "IsochronePotential",
    "JaffePotential",
    "KeplerPotential",
    "KuzminPotential",
    "LogarithmicPotential",
    "MiyamotoNagaiPotential",
    "NullPotential",
    "PlummerPotential",
    "PowerLawCutoffPotential",
    "SatohPotential",
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
from unxt import AbstractUnitSystem, Quantity, unitsystem, ustrip
from unxt.unitsystems import galactic
from xmmutablemap import ImmutableMap

import galax.typing as gt
from galax.potential._src.base import default_constants
from galax.potential._src.core import AbstractPotential
from galax.potential._src.params.core import AbstractParameter
from galax.potential._src.params.field import ParameterField

# -------------------------------------------------------------------

_burkert_const = 3 * jnp.log(jnp.asarray(2.0)) - 0.5 * jnp.pi


@final
class BurkertPotential(AbstractPotential):
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
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        m, r_s = self.m(t), self.r_s(t)
        x = jnp.linalg.vector_norm(q, axis=-1) / r_s
        xinv = 1 / x
        return -(self.constants["G"] * m / (r_s * _burkert_const)) * (
            jnp.pi
            - 2 * (1 + xinv) * jnp.atan(x).value
            + 2 * (1 + xinv) * jnp.log(1 + x)
            - (1 - xinv) * jnp.log(1 + x**2)
        )

    @partial(jax.jit, inline=True)
    def _density(
        self, q: gt.BatchQVec3, t: gt.BatchRealQScalar | gt.RealQScalar, /
    ) -> gt.BatchFloatQScalar:
        m, r_s = self.m(t), self.r_s(t)
        r = jnp.linalg.vector_norm(q, axis=-1)
        return m / (jnp.pi * _burkert_const) / ((r + r_s) * (r**2 + r_s**2))

    @partial(jax.jit, inline=True)
    def _mass(
        self, q: gt.BatchQVec3, /, t: gt.BatchRealQScalar | gt.RealQScalar
    ) -> gt.BatchFloatQScalar:
        x = jnp.linalg.vector_norm(q, axis=-1) / self.r_s(t)
        return (
            self.m(t)
            / _burkert_const
            * (-2 * jnp.atan(x) + 2 * jnp.log(1 + x) + jnp.log(1 + x**2))
        )

    # -------------------------------------------------------------------

    def rho0(self, t: gt.BatchRealQScalar | gt.RealQScalar) -> gt.BatchFloatQScalar:
        r"""Central density of the potential.

        .. math::

            m0 = \pi \rho_0 r_s^3 (3 \log(2) - \pi / 2)
        """
        return self.m(t) / (jnp.pi * self.r_s(t) ** 3 * _burkert_const)

    # -------------------------------------------------------------------
    # Constructors

    @classmethod
    def from_central_density(
        cls, rho_0: Quantity, r_s: Quantity, **kwargs: Any
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
        >>> from unxt import Quantity
        >>> from galax.potential import BurkertPotential

        >>> rho_0 = Quantity(1e6, "Msun / kpc3")
        >>> r_s = Quantity(1, "kpc")
        >>> pot = BurkertPotential.from_central_density(rho_0, r_s, units="galactic")
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
class HarmonicOscillatorPotential(AbstractPotential):
    r"""Harmonic Oscillator Potential.

    Represents an N-dimensional harmonic oscillator.

    .. math::

        \Phi(\mathbf{q}, t) = \frac{1}{2} |\omega(t) \cdot \mathbf{q}|^2

    Examples
    --------
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.HarmonicOscillatorPotential(omega=Quantity(1, "1 / Myr"),
    ...                                      units="galactic")
    >>> pot
    HarmonicOscillatorPotential(
      units=LTMAUnitSystem( ... ),
      constants=ImmutableMap({'G': ...}),
      omega=ConstantParameter( value=Quantity[...](value=f64[], unit=Unit("1 / Myr")) )
    )

    >>> q = Quantity([1.0, 0, 0], "kpc")
    >>> t = Quantity(0, "Gyr")

    >>> pot.potential(q, t)
    Quantity[...](Array(0.5, dtype=float64), unit='kpc2 / Myr2')

    >>> pot.density(q, t)
    Quantity[...](Array(1.76897707e+10, dtype=float64), unit='solMass / kpc3')

    """

    # TODO: enable omega to be a 3D vector
    omega: AbstractParameter = ParameterField(dimensions="frequency")  # type: ignore[assignment]
    """The frequency."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        # \Phi(\mathbf{q}, t) = \frac{1}{2} |\omega(t) \cdot \mathbf{q}|^2
        omega = jnp.atleast_1d(self.omega(t))
        return 0.5 * jnp.sum(jnp.square(omega * q), axis=-1)

    @partial(jax.jit, inline=True)
    def _density(
        self, _: gt.BatchQVec3, t: gt.BatchRealQScalar | gt.RealQScalar, /
    ) -> gt.BatchFloatQScalar:
        # \rho(\mathbf{q}, t) = \frac{1}{4 \pi G} \sum_i \omega_i^2
        omega = jnp.atleast_1d(self.omega(t))
        denom = 4 * jnp.pi * self.constants["G"]
        return jnp.sum(omega**2, axis=-1) / denom


# -------------------------------------------------------------------


@final
class HenonHeilesPotential(AbstractPotential):
    r"""Henon-Heiles Potential.

    This is a modified version of the [classical Henon-Heiles
    potential](https://en.wikipedia.org/wiki/Hénon-Heiles_system).

    .. math::

        \Phi * t_s^2 = \frac{1}{2} (x^2 + y^2) + \lambda (x^2 y - y^3 / 3)

    Note the timescale :math:`t_s` is introduced to convert the potential to
    specific energy, from the classical area units.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.HenonHeilesPotential(coeff=Quantity(1, "1 / kpc"),
    ...                               timescale=Quantity(1, "Myr"),
    ...                               units="galactic")
    >>> pot
    HenonHeilesPotential(
      units=LTMAUnitSystem( ... ),
      constants=ImmutableMap({'G': ...}),
      coeff=ConstantParameter( ... ),
      timescale=ConstantParameter( ... )
    )

    >>> q = Quantity([1.0, 0, 0], "kpc")
    >>> t = Quantity(0, "Gyr")
    >>> pot.potential(q, t)
    Quantity['...'](Array(0.5, dtype=float64), unit='kpc2 / Myr2')

    """

    coeff: AbstractParameter = ParameterField(dimensions="wavenumber")  # type: ignore[assignment]
    """Coefficient for the cubic terms."""

    timescale: AbstractParameter = ParameterField(dimensions="time")  # type: ignore[assignment]
    """Timescale of the potential.

    The [classical Henon-Heiles
    potential](https://en.wikipedia.org/wiki/Hénon-Heiles_system) has a
    potential with units of area.
    `galax` requires the potential to have units of specific energy, so we
    introduce a timescale parameter to convert the potential to specific
    energy.

    """

    @partial(jax.jit, inline=True)
    def _potential(
        self, q: gt.BatchQVec3, /, t: gt.BatchableRealQScalar
    ) -> gt.SpecificEnergyBatchScalar:
        ts2, coeff = self.timescale(t) ** 2, self.coeff(t)
        x2, y = q[..., 0] ** 2, q[..., 1]
        R2 = x2 + y**2
        return (R2 / 2 + coeff * (x2 * y - y**3 / 3.0)) / ts2


# -------------------------------------------------------------------


@final
class HernquistPotential(AbstractPotential):
    """Hernquist Potential."""

    m_tot: Ann[AbstractParameter, Doc("Total mass of the potential.")] = ParameterField(  # type: ignore[assignment]
        dimensions="mass"
    )

    r_s: Ann[AbstractParameter, Doc("Scale radius")] = ParameterField(
        dimensions="length"
    )  # type: ignore[assignment]

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        r = jnp.linalg.vector_norm(q, axis=-1)
        return -self.constants["G"] * self.m_tot(t) / (r + self.r_s(t))

    @partial(jax.jit, inline=True)
    def _density(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.BatchFloatQScalar:
        r_s = self.r_s(t)
        x = jnp.linalg.vector_norm(q, axis=-1) / r_s
        rho0 = self.m_tot(t) / (2 * jnp.pi * r_s**3)
        return rho0 / (x * (1 + x) ** 3)


# -------------------------------------------------------------------


@final
class IsochronePotential(AbstractPotential):
    r"""Isochrone Potential.

    .. math::

        \Phi = -\frac{G M(t)}{r_s + \sqrt{r^2 + r_s^2}}
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Total mass of the potential."""

    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    r"""Scale radius of the potential.

    The value of :math:`r_s` defines the transition between the inner, more
    harmonic oscillator-like behavior of the potential, and the outer, :math:`1
    / r` Keplerian falloff.
    """

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(  # TODO: inputs w/ units
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        r = jnp.linalg.vector_norm(q, axis=-1)
        b = self.b(t)
        return -self.constants["G"] * self.m_tot(t) / (b + jnp.sqrt(r**2 + b**2))


# -------------------------------------------------------------------


@final
class JaffePotential(AbstractPotential):
    """Jaffe Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    @partial(jax.jit, inline=True)
    def _potential(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        r = jnp.linalg.vector_norm(q, axis=-1)
        r_s = self.r_s(t)
        return -self.constants["G"] * self.m(t) / r_s * jnp.log(1 + r_s / r)


# -------------------------------------------------------------------


@final
class KeplerPotential(AbstractPotential):
    r"""The Kepler potential for a point mass.

    .. math::

        \Phi = -\frac{G M(t)}{r}
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Total mass of the potential."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(  # TODO: inputs w/ units
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        r = jnp.linalg.vector_norm(q, axis=-1)
        return -self.constants["G"] * self.m_tot(t) / r

    @partial(jax.jit, inline=True)
    def _density(
        self, q: gt.BatchQVec3, t: gt.BatchRealQScalar | gt.RealQScalar, /
    ) -> gt.BatchFloatQScalar:
        r = jnp.linalg.vector_norm(q, axis=-1)
        m = self.m_tot(t)
        pred = jnp.logical_or(  # are we at the origin with non-zero mass?
            jnp.greater(r, jnp.zeros_like(r)), jnp.equal(m, jnp.zeros_like(m))
        )
        return Quantity(
            qlax.select(
                pred,
                jnp.zeros_like(r.value),
                jnp.full_like(r.value, fill_value=jnp.inf),
            ),
            self.units["mass density"],
        )


# -------------------------------------------------------------------


@final
class KuzminPotential(AbstractPotential):
    r"""Kuzmin Potential.

    .. math::

        \Phi(x, t) = -\frac{G M(t)}{\sqrt{R^2 + (a(t) + |z|)^2}}

    See https://galaxiesbook.org/chapters/II-01.-Flattened-Mass-Distributions.html#Razor-thin-disk:-The-Kuzmin-model

    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Total mass of the potential."""

    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale length."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(
        self: "KuzminPotential", q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        return (
            -self.constants["G"]
            * self.m_tot(t)
            / jnp.sqrt(
                q[..., 0] ** 2 + q[..., 1] ** 2 + (jnp.abs(q[..., 2]) + self.a(t)) ** 2
            )
        )


# -------------------------------------------------------------------


@final
class LogarithmicPotential(AbstractPotential):
    """Logarithmic Potential."""

    v_c: AbstractParameter = ParameterField(dimensions="speed")  # type: ignore[assignment]
    r_h: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        r2 = ustrip(self.units["length"], jnp.linalg.vector_norm(q, axis=-1)) ** 2
        return (
            0.5
            * self.v_c(t) ** 2
            * jnp.log(ustrip(self.units["length"], self.r_h(t)) ** 2 + r2)
        )


# -------------------------------------------------------------------


@final
class MiyamotoNagaiPotential(AbstractPotential):
    """Miyamoto-Nagai Potential."""

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Total mass of the potential."""

    # TODO: rename
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale length in the major-axis (x-y) plane."""

    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale length in the minor-axis (x-y) plane."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(
        self: "MiyamotoNagaiPotential", q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        R2 = q[..., 0] ** 2 + q[..., 1] ** 2
        zp2 = (jnp.sqrt(q[..., 2] ** 2 + self.b(t) ** 2) + self.a(t)) ** 2
        return -self.constants["G"] * self.m_tot(t) / jnp.sqrt(R2 + zp2)


# -------------------------------------------------------------------


@final
class NullPotential(AbstractPotential):
    """Null potential, i.e. no potential.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from galax.potential import NullPotential

    >>> pot = NullPotential()
    >>> pot
    NullPotential( units=..., constants=ImmutableMap({'G': ...}) )

    >>> q = Quantity([1, 0, 0], "kpc")
    >>> t = Quantity(0, "Gyr")
    >>> pot.potential(q, t)
    Quantity[...](Array(0, dtype=int64), unit='kpc2 / Myr2')

    """

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(
        default=galactic, converter=unitsystem, static=True
    )
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(
        self,
        q: gt.BatchQVec3,
        t: gt.BatchableRealQScalar,  # noqa: ARG002
        /,
    ) -> gt.SpecificEnergyBatchScalar:
        return Quantity(  # TODO: better unit handling
            jnp.zeros(q.shape[:-1], dtype=q.dtype), galactic["specific energy"]
        )

    @partial(jax.jit, inline=True)
    def _gradient(self, q: gt.BatchQVec3, /, _: gt.RealQScalar) -> gt.BatchQVec3:
        """See ``gradient``."""
        return Quantity(  # TODO: better unit handling
            jnp.zeros(q.shape[:-1] + (3,), dtype=q.dtype), galactic["acceleration"]
        )

    @partial(jax.jit, inline=True)
    def _laplacian(self, q: gt.QVec3, /, _: gt.RealQScalar) -> gt.FloatQScalar:
        """See ``laplacian``."""
        return Quantity(  # TODO: better unit handling
            jnp.zeros(q.shape[:-1], dtype=q.dtype), galactic["frequency drift"]
        )

    @partial(jax.jit, inline=True)
    def _density(
        self, q: gt.BatchQVec3, _: gt.BatchRealQScalar | gt.RealQScalar, /
    ) -> gt.BatchFloatQScalar:
        """See ``density``."""
        return Quantity(  # TODO: better unit handling
            jnp.zeros(q.shape[:-1], dtype=q.dtype), galactic["mass density"]
        )

    @partial(jax.jit, inline=True)
    def _hessian(self, q: gt.QVec3, _: gt.RealQScalar, /) -> gt.QMatrix33:
        """See ``hessian``."""
        return Quantity(  # TODO: better unit handling
            jnp.zeros(q.shape[:-1] + (3, 3), dtype=q.dtype), galactic["frequency drift"]
        )


# -------------------------------------------------------------------


@final
class PlummerPotential(AbstractPotential):
    """Plummer Potential."""

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        r2 = jnp.linalg.vector_norm(q, axis=-1) ** 2
        return -self.constants["G"] * self.m_tot(t) / jnp.sqrt(r2 + self.b(t) ** 2)


# -------------------------------------------------------------------


@partial(jax.jit, inline=True)
def _safe_gamma_inc(a: Quantity, x: Quantity) -> Quantity:  # TODO: types
    return qsp.gammainc(a, x) * qsp.gamma(a)


@final
class PowerLawCutoffPotential(AbstractPotential):
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
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        m, a, r_c = self.m_tot(t), 0.5 * self.alpha(t), self.r_c(t)
        r = jnp.linalg.vector_norm(q, axis=-1)
        rp2 = (r / r_c) ** 2

        return (self.constants["G"] * m) * (
            (a - 1.5) * _safe_gamma_inc(1.5 - a, rp2) / (r * qsp.gamma(2.5 - a))
            + _safe_gamma_inc(1 - a, rp2) / (r_c * qsp.gamma(1.5 - a))
        )


# -------------------------------------------------------------------


@final
class SatohPotential(AbstractPotential):
    r"""SatohPotential(m, a, b, units=None, origin=None, R=None).

    Satoh potential for a flattened mass distribution.
    This is a good distribution for both disks and spheroids.

    .. math::

        \Phi = -\frac{G M}{\sqrt{R^2 + z^2 + a(a + 2\sqrt{z^2 + b^2})}}

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    a : :class:`~astropy.units.Quantity`, numeric [length]
        Scale length.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Scale height.
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    @partial(jax.jit, inline=True)
    def _potential(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        a, b = self.a(t), self.b(t)
        R2 = q[..., 0] ** 2 + q[..., 1] ** 2
        z = q[..., 2]
        term = R2 + z**2 + a * (a + 2 * jnp.sqrt(z**2 + b**2))
        return -self.constants["G"] * self.m_tot(t) / jnp.sqrt(term)


# -------------------------------------------------------------------


class StoneOstriker15Potential(AbstractPotential):
    r"""StoneOstriker15Potential(m, r_c, r_h, units=None, origin=None, R=None).

    Stone potential from `Stone & Ostriker (2015)
    <http://dx.doi.org/10.1088/2041-8205/806/2/L28>`_.

    .. math::

        \Phi = -\frac{2 G m}{\pi (r_h - r_c)} \left(
            \frac{r_h}{r} \tan^{-1}(\frac{r}{r_h})
            - \frac{r_c}{r} \tan^{-1}(\frac{r}{r_c})
            + \frac{1}{2} \log(\frac{r^2 + r_h^2}{r^2 + r_c^2})
            \right)

    Parameters
    ----------
    m_tot : :class:`~astropy.units.Quantity`, numeric [mass]
        Total mass.
    r_c : :class:`~astropy.units.Quantity`, numeric [length]
        Core radius.
    r_h : :class:`~astropy.units.Quantity`, numeric [length]
        Halo radius.
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    r_c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    r_h: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    # def __check_init__(self) -> None:
    #     _ = eqx.error_if(self.r_c, self.r_c.value >= self.r_h.value, "Core radius must be less than halo radius")   # noqa: E501, ERA001

    @partial(jax.jit, inline=True)
    def _potential(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        r_h = self.r_h(t)
        r_c = self.r_c(t)
        r = jnp.linalg.vector_norm(q, axis=-1)
        A = -2 * self.constants["G"] * self.m_tot(t) / (jnp.pi * (r_h - r_c))
        return A * (
            (r_h / r) * ustrip("rad", jnp.atan2(r, r_h))
            - (r_c / r) * ustrip("rad", jnp.atan2(r, r_c))
            + 0.5 * jnp.log((r**2 + r_h**2) / (r**2 + r_c**2))
        )


# -------------------------------------------------------------------


@final
class TriaxialHernquistPotential(AbstractPotential):
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
    >>> from unxt import Quantity
    >>> from galax.potential import TriaxialHernquistPotential

    >>> pot = TriaxialHernquistPotential(m_tot=Quantity(1e12, "Msun"),
    ...                                  r_s=Quantity(8, "kpc"), q1=1, q2=0.5,
    ...                                  units="galactic")

    >>> q = Quantity([1, 0, 0], "kpc")
    >>> t = Quantity(0, "Gyr")
    >>> pot.potential(q, t)
    Quantity[...](Array(-0.49983357, dtype=float64), unit='kpc2 / Myr2')
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Mass of the potential."""

    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale a scale length that determines the concentration of the system."""

    # TODO: move to a triaxial wrapper
    q1: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=Quantity(1.0, ""), dimensions="dimensionless"
    )
    """Scale length in the y direction divided by ``c``."""

    q2: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=Quantity(1.0, ""), dimensions="dimensionless"
    )
    """Scale length in the z direction divided by ``c``."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        converter=ImmutableMap, default=default_constants
    )

    @partial(jax.jit, inline=True)
    def _potential(  # TODO: inputs w/ units
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        r_s, q1, q2 = self.r_s(t), self.q1(t), self.q2(t)
        r_s = eqx.error_if(r_s, r_s.value <= 0, "r_s must be positive")

        rprime = jnp.sqrt(
            q[..., 0] ** 2 + (q[..., 1] / q1) ** 2 + (q[..., 2] / q2) ** 2
        )
        return -self.constants["G"] * self.m_tot(t) / (rprime + r_s)

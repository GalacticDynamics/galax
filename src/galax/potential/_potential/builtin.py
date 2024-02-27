"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "BarPotential",
    "HarmonicOscillatorPotential",
    "HenonHeilesPotential",
    "HernquistPotential",
    "IsochronePotential",
    "JaffePotential",
    "KeplerPotential",
    "KuzminPotential",
    "LogarithmicPotential",
    "LongMuraliBarPotential",
    "MiyamotoNagaiPotential",
    "NFWPotential",
    "NullPotential",
    "PlummerPotential",
    "PowerLawCutoffPotential",
    "SatohPotential",
    "StonePotential",
]

from dataclasses import KW_ONLY
from functools import partial
from typing import final

import array_api_jax_compat as xp
import equinox as eqx
import jax
from jax.scipy.special import gamma, gammainc

from .utils import converter_to_usys
from galax.potential._potential.core import AbstractPotential
from galax.potential._potential.param import AbstractParameter, ParameterField
from galax.typing import (
    BatchableFloatOrIntScalarLike,
    BatchFloatScalar,
    BatchVec3,
    FloatLike,
    FloatOrIntScalarLike,
    FloatScalar,
    Vec1,
    Vec3,
)
from galax.units import UnitSystem
from galax.utils._jax import vectorize_method
from galax.utils.dataclasses import field

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
    units: UnitSystem = eqx.field(converter=converter_to_usys, static=True)

    @partial(jax.jit)
    @vectorize_method(signature="(3),()->()")
    def _potential_energy(self, q: Vec3, /, t: FloatOrIntScalarLike) -> FloatScalar:
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
        return (self._G * self.m(t) / (2.0 * a)) * xp.log(
            (q_corot[0] - a + T_minus) / (q_corot[0] + a + T_plus),
        )


# -------------------------------------------------------------------


@final
class HarmonicOscillatorPotential(AbstractPotential):
    """Harmonic Oscillator Potential."""

    omega: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    _: KW_ONLY
    units: UnitSystem = eqx.field(converter=converter_to_usys, static=True)

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        return 0.5 * self.omega(t) ** 2 * xp.linalg.norm(q, axis=-1) ** 2


# -------------------------------------------------------------------


@final
class HenonHeilesPotential(AbstractPotential):
    """Henon-Heiles Potential."""

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        return 0.5 * (
            q[..., 0] ** 2
            + q[..., 1] ** 2
            + 2 * q[..., 0] ** 2 * q[..., 1]
            - 2 / 3.0 * q[..., 1] ** 3
        )


# -------------------------------------------------------------------


@final
class HernquistPotential(AbstractPotential):
    """Hernquist Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    _: KW_ONLY
    units: UnitSystem = eqx.field(converter=converter_to_usys, static=True)

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        r = xp.linalg.vector_norm(q, axis=-1)
        return -self._G * self.m(t) / (r + self.c(t))


# -------------------------------------------------------------------


@final
class IsochronePotential(AbstractPotential):
    """Isochrone Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    _: KW_ONLY
    units: UnitSystem = eqx.field(converter=converter_to_usys, static=True)

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        r = xp.linalg.vector_norm(q, axis=-1)
        b = self.b(t)
        return -self._G * self.m(t) / (b + xp.sqrt(r**2 + b**2))


# -------------------------------------------------------------------


@final
class JaffePotential(AbstractPotential):
    """Jaffe Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        r = xp.linalg.vector_norm(q, axis=-1)
        c = self.c(t)
        return -self._G * self.m(t) / c * xp.log(r / (r + c))


# -------------------------------------------------------------------


@final
class KeplerPotential(AbstractPotential):
    r"""The Kepler potential for a point mass.

    .. math::
        \Phi = -\frac{G M(t)}{r}
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    _: KW_ONLY
    units: UnitSystem = eqx.field(converter=converter_to_usys, static=True)

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        r = xp.linalg.vector_norm(q, axis=-1)
        return -self._G * self.m(t) / r


# -------------------------------------------------------------------


@final
class KuzminPotential(AbstractPotential):
    """Kuzmin Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        rp = xp.sqrt(
            q[..., 0] ** 2 + q[..., 1] ** 2 + (xp.abs(q[..., 2]) + self.a(t)) ** 2
        )
        return -self._G * self.m(t) / rp


# -------------------------------------------------------------------


@final
class LogarithmicPotential(AbstractPotential):
    """Logarithmic Potential."""

    v_c: AbstractParameter = ParameterField(dimensions="speed")  # type: ignore[assignment]
    r_h: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    q1: AbstractParameter = ParameterField(dimensions="dimensionless")  # type: ignore[assignment]
    q2: AbstractParameter = ParameterField(dimensions="dimensionless")  # type: ignore[assignment]
    q3: AbstractParameter = ParameterField(dimensions="dimensionless")  # type: ignore[assignment]
    phi: AbstractParameter = ParameterField(dimensions="angle")  # type: ignore[assignment]

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        r2 = (
            (q[..., 0] / self.q1(t)) ** 2
            + (q[..., 1] / self.q2(t)) ** 2
            + (q[..., 2] / self.q3(t)) ** 2
        )
        return 0.5 * self.v_c(t) ** 2 * xp.log(self.r_h(t) ** 2 + r2)


# -------------------------------------------------------------------


@final
class LongMuraliBarPotential(AbstractPotential):
    """Long & Murali Bar Potential.

    A simple, triaxial model for a galaxy bar. This is a softened “needle”
    density distribution with an analytic potential form. See Long & Murali
    (1992) for details.
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    alpha: AbstractParameter = ParameterField(dimensions="angle")  # type: ignore[assignment]

    @partial(jax.jit)
    def _potential_energy(self, q: Vec3, /, t: FloatOrIntScalarLike) -> FloatScalar:
        m = self.m(t)
        a, b, c = self.a(t), self.b(t), self.c(t)
        alpha = self.alpha(t)

        x = q[..., 0] * xp.cos(alpha) + q[..., 1] * xp.sin(alpha)
        y = -q[..., 0] * xp.sin(alpha) + q[..., 1] * xp.cos(alpha)
        z = q[..., 2]

        Tm = xp.sqrt((a - x) ** 2 + y**2 + (b + xp.sqrt(c**2 + z**2)) ** 2)
        Tp = xp.sqrt((a + x) ** 2 + y**2 + (b + xp.sqrt(c**2 + z**2)) ** 2)

        return self._G * m / (2 * a) * (xp.log(x - a + Tm) - xp.log(x + a + Tp))


# -------------------------------------------------------------------


@final
class MiyamotoNagaiPotential(AbstractPotential):
    """Miyamoto-Nagai Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    _: KW_ONLY
    units: UnitSystem = eqx.field(converter=converter_to_usys, static=True)

    @partial(jax.jit)
    @vectorize_method(signature="(3),()->()")
    def _potential_energy(self, q: Vec3, /, t: FloatOrIntScalarLike) -> FloatScalar:
        R2 = q[0] ** 2 + q[1] ** 2
        return (
            -self._G
            * self.m(t)
            / xp.sqrt(R2 + xp.square(xp.sqrt(q[2] ** 2 + self.b(t) ** 2) + self.a(t)))
        )


# -------------------------------------------------------------------


@final
class NFWPotential(AbstractPotential):
    """NFW Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    _: KW_ONLY
    softening_length: FloatLike = field(default=0.001, static=True, dimensions="length")
    units: UnitSystem = eqx.field(converter=converter_to_usys, static=True)

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        v_h2 = -self._G * self.m(t) / self.r_s(t)
        r2 = q[..., 0] ** 2 + q[..., 1] ** 2 + q[..., 2] ** 2
        m = xp.sqrt(r2 + self.softening_length) / self.r_s(t)
        return v_h2 * xp.log(1.0 + m) / m


# -------------------------------------------------------------------


@final
class NullPotential(AbstractPotential):
    """Null potential, i.e. no potential."""

    _: KW_ONLY
    units: UnitSystem = eqx.field(converter=converter_to_usys, static=True)

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        return xp.zeros(q.shape[:-1], dtype=q.dtype)


# -------------------------------------------------------------------


@final
class PlummerPotential(AbstractPotential):
    """Plummer Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        r2 = xp.linalg.vector_norm(q, axis=-1) ** 2
        return -self._G * self.m(t) / xp.sqrt(r2 + self.b(t) ** 2)


# -------------------------------------------------------------------


def safe_gamma_inc(a: Vec1, x: Vec1) -> Vec1:
    A = 1.0
    B = 0.0

    if a > 0:
        return gammainc(a, x) * gamma(a)

    N = xp.ceil(-a)

    for n in range(N):
        A = A * (a + n)

        tmp = 1.0
        for m in range(N - 1, n, -1):
            tmp = tmp * (a + m)
        B = B + xp.pow(x, a + n) * xp.exp(-x) * tmp

    return (B + gammainc(a + N, x) * gamma(a + N)) / A


@final
class PowerLawCutoffPotential(AbstractPotential):
    r"""A spherical power-law density profile with an exponential cutoff.

    The power law index must be ``0 <= alpha < 3``.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Total mass.
    alpha : numeric
        Power law index. Must satisfy: ``alpha < 3``
    r_c : :class:`~astropy.units.Quantity`, numeric [length]
        Cutoff radius.
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    alpha: AbstractParameter = ParameterField(dimensions="dimensionless")  # type: ignore[assignment]
    r_c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        G = self._G
        m = self.m(t)
        alpha = self.alpha(t)
        r_c = self.r_c(t)
        r2 = q[..., 0] ** 2 + q[..., 1] ** 2 + q[..., 2] ** 2

        if r2 == 0:  # TODO: handle this in a differentiable way
            return -xp.asarray(xp.inf)

        tmp_0 = -(1.0 / 2.0) * alpha
        tmp_1 = tmp_0
        tmp_2 = tmp_1 + 1.5
        tmp_4 = r2 / r_c**2
        tmp_5 = G * m
        tmp_6 = (
            tmp_5 * safe_gamma_inc(tmp_2, tmp_4) / (xp.sqrt(r2) * gamma(tmp_1 + 2.5))
        )
        return (
            -tmp_0 * tmp_6
            - 3.0 / 2.0 * tmp_6
            + tmp_5 * safe_gamma_inc(tmp_1 + 1, tmp_4) / (r_c * gamma(tmp_2))
        )


# -------------------------------------------------------------------


@final
class SatohPotential(AbstractPotential):
    r"""SatohPotential(m, a, b, units=None, origin=None, R=None).

    Satoh potential for a flattened mass distribution.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    a : :class:`~astropy.units.Quantity`, numeric [length]
        Scale length.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Scale height.
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        a, b = self.a(t), self.b(t)
        R2 = q[..., 0] ** 2 + q[..., 1] ** 2
        z = q[..., 2]
        term = R2 + z**2 + a * (a + 2 * xp.sqrt(z**2 + b**2))
        return -self._G * self.m(t) / xp.sqrt(term)


# -------------------------------------------------------------------


@final
class StonePotential(AbstractPotential):
    r"""StonePotential(m, r_c, r_h, units=None, origin=None, R=None).

    Stone potential from `Stone & Ostriker (2015)
    <http://dx.doi.org/10.1088/2041-8205/806/2/L28>`_.

    Parameters
    ----------
    m_tot : :class:`~astropy.units.Quantity`, numeric [mass]
        Total mass.
    r_c : :class:`~astropy.units.Quantity`, numeric [length]
        Core radius.
    r_h : :class:`~astropy.units.Quantity`, numeric [length]
        Halo radius.
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    r_c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    r_h: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        r_h = self.r_h(t)
        r_c = self.r_c(t)
        r = xp.linalg.vector_norm(q, axis=-1)
        A = -2 * self._G * self.m(t) / (xp.pi * (r_h - r_c))
        return A * (
            (r_h / r) * xp.atan(r / r_h)
            - (r_c / r) * xp.atan(r / r_c)
            + 0.5 * (xp.log(r**2 + r_h**2) - xp.log(r**2 + r_c**2))
        )

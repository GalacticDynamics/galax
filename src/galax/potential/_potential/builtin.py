"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "BarPotential",
    "HernquistPotential",
    "IsochronePotential",
    "KeplerPotential",
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

import array_api_jax_compat as xp
from jax_quantity import Quantity

from galax.potential._potential.core import AbstractPotential
from galax.potential._potential.param import AbstractParameter, ParameterField
from galax.typing import (
    BatchableRealScalarLike,
    BatchFloatScalar,
    BatchVec3,
    FloatLike,
    FloatScalar,
    RealScalarLike,
    Vec3,
)
from galax.units import UnitSystem, unitsystem
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
    units: UnitSystem = eqx.field(converter=unitsystem, static=True)

    # TODO: inputs w/ units
    @partial(jax.jit)
    @vectorize_method(signature="(3),()->()")
    def _potential_energy(self, q: Vec3, /, t: RealScalarLike) -> FloatScalar:
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
class HernquistPotential(AbstractPotential):
    """Hernquist Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    _: KW_ONLY
    units: UnitSystem = eqx.field(converter=unitsystem, static=True)

    @partial(jax.jit)
    def _potential_energy(  # TODO: inputs w/ units
        self, q: BatchVec3, /, t: BatchableRealScalarLike
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
    units: UnitSystem = eqx.field(converter=unitsystem, static=True)

    @partial(jax.jit)
    def _potential_energy(  # TODO: inputs w/ units
        self, q: BatchVec3, /, t: BatchableRealScalarLike
    ) -> BatchFloatScalar:
        r = xp.linalg.vector_norm(q, axis=-1)
        b = self.b(t)
        return -self._G * self.m(t) / (b + xp.sqrt(r**2 + b**2))


# -------------------------------------------------------------------


@final
class KeplerPotential(AbstractPotential):
    r"""The Kepler potential for a point mass.

    .. math::
        \Phi = -\frac{G M(t)}{r}
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    _: KW_ONLY
    units: UnitSystem = eqx.field(converter=unitsystem, static=True)

    @partial(jax.jit)
    def _potential_energy(  # TODO: inputs w/ units
        self, q: BatchVec3, /, t: BatchableRealScalarLike
    ) -> BatchFloatScalar:
        r = xp.linalg.vector_norm(q, axis=-1)
        return -self._G * self.m(t) / r


# -------------------------------------------------------------------


@final
class MiyamotoNagaiPotential(AbstractPotential):
    """Miyamoto-Nagai Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    _: KW_ONLY
    units: UnitSystem = eqx.field(converter=unitsystem, static=True)

    # TODO: inputs w/ units
    @partial(jax.jit)
    @vectorize_method(signature="(3),()->()")
    def _potential_energy(self, q: Vec3, /, t: RealScalarLike) -> FloatScalar:
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
    units: UnitSystem = eqx.field(converter=unitsystem, static=True)

    @partial(jax.jit)
    def _potential_energy(  # TODO: inputs w/ units
        self, q: BatchVec3, /, t: BatchableRealScalarLike
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
    units: UnitSystem = eqx.field(converter=unitsystem, static=True)

    @partial(jax.jit)
    def _potential_energy(  # TODO: inputs w/ units
        self,
        q: BatchVec3,
        /,
        t: BatchableRealScalarLike,  # noqa: ARG002
    ) -> BatchFloatScalar:
        return xp.zeros(q.shape[:-1], dtype=q.dtype)


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

    units : :class:`~galax.units.UnitSystem`, keyword-only
        The unit system to use for the potential.  This parameter accepts a
        :class:`~galax.units.UnitSystem` or anything that can be converted to a
        :class:`~galax.units.UnitSystem` using :func:`~galax.units.unitsystem`.

    Examples
    --------
    >>> from jax_quantity import Quantity
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
    units: UnitSystem = eqx.field(converter=unitsystem, static=True)
    """The unit system to use for the potential."""

    @partial(jax.jit)
    def _potential_energy(  # TODO: inputs w/ units
        self, q: BatchVec3, /, t: BatchableRealScalarLike
    ) -> BatchFloatScalar:
        c, q1, q2 = self.c(t), self.q1(t), self.q2(t)
        c = eqx.error_if(c, c <= 0, "c must be positive")

        rprime = xp.sqrt(q[..., 0] ** 2 + (q[..., 1] / q1) ** 2 + (q[..., 2] / q2) ** 2)
        return -self._G * self.m(t) / (rprime + c)

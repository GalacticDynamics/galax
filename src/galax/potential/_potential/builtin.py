"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "BarPotential",
    "HernquistPotential",
    "IsochronePotential",
    "KeplerPotential",
    "MiyamotoNagaiPotential",
    "NFWPotential",
    "NullPotential",
]

from dataclasses import KW_ONLY
from typing import final

import astropy.units as u
import jax.experimental.array_api as xp

from galax.potential._potential.core import AbstractPotential
from galax.potential._potential.param import AbstractParameter, ParameterField
from galax.typing import (
    BatchableFloatOrIntScalarLike,
    BatchFloatScalar,
    BatchVec3,
    FloatLike,
    FloatOrIntScalarLike,
    FloatScalar,
    Vec3,
)
from galax.utils import partial_jit, vectorize_method
from galax.utils.dataclasses import field

mass = u.get_physical_type("mass")
length = u.get_physical_type("length")
frequency = u.get_physical_type("frequency")

# -------------------------------------------------------------------


@final
class BarPotential(AbstractPotential):
    """Rotating bar potentil, with hard-coded rotation.

    Eq 8a in https://articles.adsabs.harvard.edu/pdf/1992ApJ...397...44L
    Rz according to https://en.wikipedia.org/wiki/Rotation_matrix
    """

    m: AbstractParameter = ParameterField(dimensions=mass)  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions=length)  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions=length)  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(dimensions=length)  # type: ignore[assignment]
    Omega: AbstractParameter = ParameterField(dimensions=frequency)  # type: ignore[assignment]

    @partial_jit()
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
class HernquistPotential(AbstractPotential):
    """Hernquist Potential."""

    m: AbstractParameter = ParameterField(dimensions=mass)  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(dimensions=length)  # type: ignore[assignment]

    @partial_jit()
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        r = xp.linalg.vector_norm(q, axis=-1)
        return -self._G * self.m(t) / (r + self.c(t))


# -------------------------------------------------------------------


@final
class IsochronePotential(AbstractPotential):
    """Isochrone Potential."""

    m: AbstractParameter = ParameterField(dimensions=mass)  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions=length)  # type: ignore[assignment]

    @partial_jit()
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
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

    m: AbstractParameter = ParameterField(dimensions=mass)  # type: ignore[assignment]

    @partial_jit()
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        r = xp.linalg.vector_norm(q, axis=-1)
        return -self._G * self.m(t) / r


# -------------------------------------------------------------------


@final
class MiyamotoNagaiPotential(AbstractPotential):
    """Miyamoto-Nagai Potential."""

    m: AbstractParameter = ParameterField(dimensions=mass)  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions=length)  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions=length)  # type: ignore[assignment]

    @partial_jit()
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        x, y, z = q[..., 0], q[..., 1], q[..., 2]
        R2 = x**2 + y**2
        return (
            -self._G
            * self.m(t)
            / xp.sqrt(R2 + xp.square(xp.sqrt(z**2 + self.b(t) ** 2) + self.a(t)))
        )


# -------------------------------------------------------------------


@final
class NFWPotential(AbstractPotential):
    """NFW Potential."""

    m: AbstractParameter = ParameterField(dimensions=mass)  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions=length)  # type: ignore[assignment]
    _: KW_ONLY
    softening_length: FloatLike = field(default=0.001, static=True, dimensions=length)

    @partial_jit()
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

    @partial_jit()
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        return xp.zeros(q.shape[:-1], dtype=q.dtype)

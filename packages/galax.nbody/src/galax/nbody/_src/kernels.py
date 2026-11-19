"""Softening kernels for N-body forces.

The pair contribution to target i's acceleration is
``G * m_j * force_factor(r2, h_ij) * (x_j - x_i)`` and to its potential
``G * m_j * potential(r2, h_ij)``.
"""

__all__ = [
    "AbstractSofteningKernel",
    "DehnenK1Kernel",
    "DehnenK2Kernel",
    "NewtonianKernel",
    "PlummerKernel",
    "SplineKernel",
    "max_softening",
    "plummer_force_factor",
    "plummer_potential",
]

import abc

from jaxtyping import Array

import equinox as eqx
import jax
import jax.numpy as jnp


def plummer_force_factor(r2: Array, h: Array) -> Array:
    """Return the Plummer factor ``(r2 + h^2)^(-3/2)`` that multiplies ``dx``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from galax.nbody._src.kernels import plummer_force_factor
    >>> float(plummer_force_factor(jnp.array(3.0), jnp.array(1.0)))
    0.125

    """
    return jax.lax.rsqrt(r2 + h**2) ** 3


def plummer_potential(r2: Array, h: Array) -> Array:
    """Return the Plummer potential ``-(r2 + h^2)^(-1/2)`` per unit ``G m``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from galax.nbody._src.kernels import plummer_potential
    >>> float(plummer_potential(jnp.array(3.0), jnp.array(1.0)))
    -0.5

    """
    return -jax.lax.rsqrt(r2 + h**2)


def max_softening(h_i: Array, h_j: Array) -> Array:
    """Return the larger of two softening lengths.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from galax.nbody._src.kernels import max_softening
    >>> float(max_softening(jnp.array(0.1), jnp.array(0.3)))
    0.3

    """
    return jnp.maximum(h_i, h_j)


class AbstractSofteningKernel(eqx.Module):
    """Base class for softening kernels."""

    @abc.abstractmethod
    def force_factor(self, r2: Array, h: Array) -> Array:
        """Return the factor that multiplies ``x_j - x_i``."""

    @abc.abstractmethod
    def potential(self, r2: Array, h: Array) -> Array:
        """Return the potential per unit ``G m``."""

    def combine(self, h_i: Array, h_j: Array) -> Array:
        """Return the pair softening length, by default the larger one."""
        return max_softening(h_i, h_j)


class PlummerKernel(AbstractSofteningKernel):
    """Plummer softening kernel."""

    def force_factor(self, r2: Array, h: Array) -> Array:
        """Return the Plummer force factor."""
        return plummer_force_factor(r2, h)

    def potential(self, r2: Array, h: Array) -> Array:
        """Return the Plummer potential."""
        return plummer_potential(r2, h)


class NewtonianKernel(AbstractSofteningKernel):
    """Unsoftened gravity; ``h`` is ignored."""

    def force_factor(self, r2: Array, h: Array) -> Array:
        """Return ``r^-3``."""
        del h
        return jax.lax.rsqrt(r2) ** 3

    def potential(self, r2: Array, h: Array) -> Array:
        """Return ``-1/r``."""
        del h
        return -jax.lax.rsqrt(r2)


class DehnenK1Kernel(AbstractSofteningKernel):
    """Dehnen (2001) K1 kernel; Plummer with an ``h^2`` correction."""

    def force_factor(self, r2: Array, h: Array) -> Array:
        """Return ``d^-3/2 + 3/2 h^2 d^-5/2`` with ``d = r2 + h^2``."""
        h2 = h * h
        s = jax.lax.rsqrt(r2 + h2)
        s3 = s**3
        return s3 + 1.5 * h2 * s3 * s * s

    def potential(self, r2: Array, h: Array) -> Array:
        """Return ``-d^-1/2 - 1/2 h^2 d^-3/2``."""
        h2 = h * h
        s = jax.lax.rsqrt(r2 + h2)
        return -s - 0.5 * h2 * s**3


class DehnenK2Kernel(AbstractSofteningKernel):
    """Dehnen (2001) K2 kernel; Plummer with ``h^2`` and ``h^4`` corrections."""

    def force_factor(self, r2: Array, h: Array) -> Array:
        """Return ``d^-3/2 + 3/2 h^2 d^-5/2 + 15/8 h^4 d^-7/2``."""
        h2 = h * h
        s = jax.lax.rsqrt(r2 + h2)
        s2, s3 = s * s, s**3
        return s3 * (1.0 + h2 * s2 * (1.5 + 1.875 * h2 * s2))

    def potential(self, r2: Array, h: Array) -> Array:
        """Return ``-d^-1/2 - 1/2 h^2 d^-3/2 - 3/8 h^4 d^-5/2``."""
        h2 = h * h
        s = jax.lax.rsqrt(r2 + h2)
        s2 = s * s
        return -s * (1.0 + h2 * s2 * (0.5 + 0.375 * h2 * s2))


def _spline_q(r2: Array, h: Array) -> tuple[Array, Array, Array, Array]:
    """Return ``(r, q, 1/h, inside)`` with gradient-safe square roots."""
    one, zero = jnp.ones_like(r2), jnp.zeros_like(r2)
    pos = r2 > 0
    r = jnp.where(pos, jnp.sqrt(jnp.where(pos, r2, one)), zero)
    inside = r < h
    hinv = 1.0 / jnp.where(inside, h, jnp.ones_like(h))
    return r, r * hinv, hinv, inside


class SplineKernel(AbstractSofteningKernel):
    """Monaghan cubic-spline kernel (as in Gadget); ``h`` is the support radius.

    Newtonian for ``r >= h``. Plummer-equivalent softening is about ``h / 2.8``.
    """

    def force_factor(self, r2: Array, h: Array) -> Array:
        """Return the spline force factor."""
        r, q, hinv, inside = _spline_q(r2, h)
        h3 = hinv**3
        qs = jnp.where(q > 0, q, jnp.ones_like(q))
        inner = 32.0 / 3 + q * q * (32.0 * q - 38.4)
        outer = 64.0 / 3 + q * (-48.0 + q * (38.4 - 32.0 / 3 * q)) - 1 / 15 / qs**3
        soft = h3 * jnp.where(q <= 0.5, inner, outer)
        rs = jnp.where(inside, jnp.ones_like(r), r)
        return jnp.where(inside, soft, 1.0 / rs**3)

    def potential(self, r2: Array, h: Array) -> Array:
        """Return the spline potential."""
        r, q, hinv, inside = _spline_q(r2, h)
        q2 = q * q
        qs = jnp.where(q > 0, q, jnp.ones_like(q))
        inner = -2.8 + q2 * (16.0 / 3 + q2 * (6.4 * q - 9.6))
        outer = (
            -3.2
            + 1 / 15 / qs
            + q2 * (32.0 / 3 + q * (-16.0 + q * (9.6 - 32.0 / 15 * q)))
        )
        soft = hinv * jnp.where(q <= 0.5, inner, outer)
        rs = jnp.where(inside, jnp.ones_like(r), r)
        return jnp.where(inside, soft, -1.0 / rs)

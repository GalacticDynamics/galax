"""Softening kernels for N-body forces.

The pair contribution to target i's acceleration is
``G * m_j * force_factor(r2, h_ij) * (x_j - x_i)`` and to its potential
``G * m_j * potential(r2, h_ij)``.
"""

__all__ = [
    "AbstractSofteningKernel",
    "PlummerKernel",
    "max_softening",
    "plummer_force_factor",
    "plummer_potential",
]

from jaxtyping import Array

import equinox as eqx
import jax


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
    return jax.numpy.maximum(h_i, h_j)


class AbstractSofteningKernel(eqx.Module):
    """Base class for softening kernels."""

    def force_factor(self, r2: Array, h: Array) -> Array:
        """Return the factor that multiplies ``x_j - x_i``."""
        raise NotImplementedError

    def potential(self, r2: Array, h: Array) -> Array:
        """Return the potential per unit ``G m``."""
        raise NotImplementedError

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

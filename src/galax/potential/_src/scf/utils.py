"""Utility Functions."""

from functools import partial
from typing import TypeAlias, TypeVar, cast

import jax
from jaxtyping import ArrayLike, Shaped

import quaxed.numpy as jnp

import galax._custom_types as gt
from galax.potential._src.builtin.multipole import compute_Ylm

BatchableIntSz0: TypeAlias = Shaped[gt.IntSz0, "*#batch"]

T = TypeVar("T", bound=ArrayLike)


@partial(jax.jit)
@partial(jax.numpy.vectorize, signature="(3)->(3)")
def cartesian_to_spherical(xyz: gt.FloatSz3, /) -> gt.FloatSz3:
    """Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    xyz : Array[float, (3,)]
        Cartesian coordinates in the form (x, y, z).

    Returns
    -------
    r_theta_phi : Array[float, (3,)]
        Spherical radius.
        Inclination (polar) angle in [0, pi] from North to South pole.
        Azimuthal angle in [-pi, pi]
    """
    r = jnp.sqrt(jnp.sum(xyz**2, axis=0))  # spherical radius
    # TODO: this is a hack to avoid the ambiguity at r==0. This should be done better.
    theta = jax.lax.select(
        r == 0, jnp.zeros_like(r), jnp.arccos(xyz[2] / r)
    )  # inclination angle
    phi = jnp.arctan2(xyz[1], xyz[0])  # azimuthal angle
    return jnp.array([r, theta, phi])

def psi_of_r(r: T) -> T:
    r""":math:`\psi(r) = (r-1)/(r+1)`.

    Equation 9 of Lowing et al. (2011).
    """
    return cast("T", (r - 1.0) / (r + 1.0))


# =============================================================================


@partial(jax.jit, static_argnames=("m_max",))
def real_Ylm(
    theta: gt.SzAny, l: BatchableIntSz0, m: BatchableIntSz0, m_max: int = 100
) -> gt.SzAny:
    r"""Get the spherical harmonic :math:`Y_{lm}(\theta)` of the polar angle.

    This is different than the scipy (and thus JAX) convention, which is
    :math:`Y_{lm}(\theta, \phi)`.
    Note that scipy also uses the opposite convention for theta, phi where
    theta is the azimuthal angle and phi is the polar angle.

    Parameters
    ----------
    theta : Array[float, (*shape, N)]
        Polar angle in [0, pi].
    l, m : int | Array[int, ()]
        Spherical harmonic terms. l in [0,lmax], m in [0,l].
    m_max : int, optional
        Maximum order of the spherical harmonic expansion.

    Returns
    -------
    Array[float, (l, m, N)]
        Spherical harmonic. The shape is batched using the vectorization signature
        ``(n),(),()->(n)``.
    """
    # TODO: raise an error if m > m_max
    _real_Ylm, _ = compute_Ylm(l, m, theta=theta, phi=jnp.zeros_like(theta), l_max=m_max)
    return _real_Ylm

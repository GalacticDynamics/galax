"""Self-Consistent Field Potential."""

__all__: list[str] = []

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .coeffs_helper import normalization_Knl
from .utils import psi_of_r
from spexial import eval_gegenbauers
import galax._custom_types as gt


def rho_nl(n: gt.IntSz0, l: gt.IntSz0, s: gt.FloatSz0,
) -> gt.FloatSz0:
    r"""Radial density expansion terms.

    Parameters
    ----------
    s : Array[float, (n,)]
        Scaled radius :math:`r/r_s`.
    n : int
        Radial expansion term.
    l : int
        Spherical harmonic term.

    Returns
    -------
    Array[float, (n,)]
    """
    return (
        jnp.sqrt(4 * jnp.pi)
        * (normalization_Knl(n=n, l=l) / (2 * jnp.pi))
        * (s**l / (s * (1 + s) ** (2 * l + 3)))
        * eval_gegenbauers(n, 2 * l + 1.5, psi_of_r(s))
    )

rho_nl_jit_vec = jax.jit(
    jax.vmap( jax.vmap(rho_nl, in_axes=(None, 0, None),), in_axes=(None, None, 0)), static_argnames="n"
)

# ======================================================================


def phi_nl(n: gt.IntSz0, l: gt.IntSz0, s: gt.FloatSz0,
) -> gt.FloatSz0:
    r"""Angular density expansion terms.

    Parameters
    ----------
    n : int
        Max Radial expansion term.
    l : int
        Spherical harmonic term.
    s : Float
        Scaled radius :math:`r/r_s`.

    Returns
    -------
    Array[float, (n + 1,)]

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> phi_nl(0.5, 1, 1)
    Array(0.5, dtype=float32)
    >>> phi_nl(jnp.array([0.5, 0.5]), 1, 1)
    Array([0.5, 0.5], dtype=float32)
    """
    return (
        -jnp.sqrt(4 * jnp.pi)
        * (s**l / (1.0 + s) ** (2 * l + 1))
        * eval_gegenbauers(n, 2 * l + 1.5, psi_of_r(s))
    )

phi_nl_jit_vec = jax.jit(
    jax.vmap( jax.vmap(phi_nl, in_axes=(None, 0, None),), in_axes=(None, None, 0)), static_argnames="n"
)
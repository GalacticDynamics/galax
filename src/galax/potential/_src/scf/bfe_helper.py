"""Self-Consistent Field Potential."""

__all__: list[str] = []

from functools import partial

import jax
from jaxtyping import Array, Float

from .coeffs_helper import normalization_Knl
from .utils import psi_of_r
from spexial import eval_gegenbauer
import galax._custom_types as gt


@jax.jit
def rho_nl(
    s: Float[Array, "N"], n: int, l: int, *,
) -> Float[Array, "N"]:
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
        * eval_gegenbauer(n, 2 * l + 1.5, psi_of_r(s))
    )


# ======================================================================


@jax.jit
def phi_nl(
    s: Float[Array, "samples"], n: IntLike, l: IntLike,
) -> Float[Array, "samples"]:
    r"""Angular density expansion terms.

    Parameters
    ----------
    s : Array[float, (n_samples,)]
        Scaled radius :math:`r/r_s`.
    n : int
        Radial expansion term.
    l : int
        Spherical harmonic term.

    Returns
    -------
    Array[float, (n_samples,)]

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
        * eval_gegenbauer(n, 2 * l + 1.5, psi_of_r(s))
    )


phi_nl_vec = jnp.vectorize(phi_nl, signature="(n),(),()->(n)", excluded=(3,))

phi_nl_grad = jax.jit(jnp.vectorize(jax.grad(phi_nl, argnums=0), excluded=(3,)))
r"""Derivative :math:`\frac{d}{dr} \phi_{nl}`."""

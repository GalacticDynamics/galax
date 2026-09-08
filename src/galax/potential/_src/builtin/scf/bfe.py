"""Self-Consistent Field basis functions."""

__all__ = ["phi_nl", "rho_nl"]

import functools as ft

from jaxtyping import Array, Float

import jax

import quaxed.numpy as jnp

from .gegenbauer import gegenbauer_all

SQRT_FOURPI = 3.544907701811031
"""``sqrt(4 * pi)``, matching the literal in gala's ``bfe_helper.cpp``."""


def _nl_axes(
    nmax: int, lmax: int, s: Float[Array, "*batch"], /
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Broadcastable ``n``, ``l`` and Gegenbauer table for the given ``s``."""
    nbatch = jnp.ndim(s)
    ls = jnp.arange(lmax + 1, dtype=s.dtype)
    # n gets one axis for l plus one per batch axis; l gets one per batch axis.
    n = jnp.expand_dims(
        jnp.arange(nmax + 1, dtype=s.dtype), tuple(range(1, 2 + nbatch))
    )
    l = jnp.expand_dims(ls, tuple(range(1, 1 + nbatch)))
    cn = gegenbauer_all(nmax, 2 * ls + 1.5, (s - 1) / (s + 1))
    return n, l, cn


@ft.partial(jax.jit, static_argnums=(0, 1))
def phi_nl(
    nmax: int, lmax: int, s: Float[Array, "*batch"], /
) -> Float[Array, "{nmax}+1 {lmax}+1 *batch"]:
    r"""Radial potential expansion terms.

    $$ \phi_{nl}(s) = -\sqrt{4\pi} \frac{s^l}{(1+s)^{2l+1}} C_n^{2l+3/2}(\xi) $$

    with $\xi = (s-1)/(s+1)$.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from galax.potential.scf import phi_nl

    The $n = l = 0$ term is the Hernquist profile up to normalization:

    >>> bool(jnp.allclose(phi_nl(0, 0, jnp.asarray(1.0)),
    ...                   -3.544907701811031 / 2))
    True

    """
    _, l, cn = _nl_axes(nmax, lmax, s)
    prefactor = -SQRT_FOURPI * s**l / (1 + s) ** (2 * l + 1)
    return prefactor * cn


@ft.partial(jax.jit, static_argnums=(0, 1))
def rho_nl(
    nmax: int, lmax: int, s: Float[Array, "*batch"], /
) -> Float[Array, "{nmax}+1 {lmax}+1 *batch"]:
    r"""Radial density expansion terms.

    $$ \rho_{nl}(s) = \sqrt{4\pi} \frac{K_{nl}}{2\pi}
                      \frac{s^l}{s(1+s)^{2l+3}} C_n^{2l+3/2}(\xi) $$

    with $K_{nl} = \frac{1}{2}n(n+4l+3) + (l+1)(2l+1)$.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from galax.potential.scf import rho_nl

    >>> rho_nl(0, 0, jnp.asarray(1.0)).shape
    (1, 1)

    """
    n, l, cn = _nl_axes(nmax, lmax, s)
    knl = 0.5 * n * (n + 4 * l + 3) + (l + 1) * (2 * l + 1)
    prefactor = SQRT_FOURPI * (knl / (2 * jnp.pi)) * s**l / (s * (1 + s) ** (2 * l + 3))
    return prefactor * cn  # type: ignore[no-any-return]

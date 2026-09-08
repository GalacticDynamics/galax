"""Gegenbauer (ultraspherical) polynomials."""

__all__ = ["gegenbauer_all"]

import functools as ft

from jaxtyping import Array, Float

import jax

import quaxed.numpy as jnp


@ft.partial(jax.jit, static_argnums=(0,))
def gegenbauer_all(
    nmax: int, alpha: Float[Array, "L"], x: Float[Array, "*batch"], /
) -> Float[Array, "{nmax}+1 L *batch"]:
    r"""Gegenbauer polynomials $C_n^\alpha(x)$ for every $n \le n_{max}$.

    The three-term recurrence

    $$ n C_n^\alpha(x) = 2x(n + \alpha - 1) C_{n-1}^\alpha(x)
                         - (n + 2\alpha - 2) C_{n-2}^\alpha(x) $$

    is stepped once, carrying every $\alpha$ and every $x$ together, so the
    whole table costs $O(n_{max})$ rather than the $O(n_{max}^2)$ of evaluating
    each order independently.

    Parameters
    ----------
    nmax
        Maximum order. Static: it sets the length of the leading axis.
    alpha
        Ultraspherical parameters, one per output column.
    x
        Argument(s), of any batch shape.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from galax.potential.scf import gegenbauer_all

    $C_0 = 1$ and $C_1 = 2 \alpha x$:

    >>> gegenbauer_all(1, jnp.asarray([1.5]), jnp.asarray([0.5]))
    Array([[[1. ]],
           [[1.5]]], dtype=float64)

    """
    alpha, x = jnp.asarray(alpha), jnp.asarray(x)
    # Give alpha a trailing axis per batch axis of x, so the two broadcast.
    a = jnp.expand_dims(alpha, tuple(range(1, 1 + jnp.ndim(x))))

    c0 = jnp.ones(jnp.broadcast_shapes(jnp.shape(a), jnp.shape(x)))
    if nmax == 0:
        return c0[None]  # type: ignore[no-any-return]
    c1 = 2 * a * x * jnp.ones_like(c0)

    def step(
        carry: tuple[Float[Array, "..."], Float[Array, "..."]], n: Float[Array, ""]
    ) -> tuple[tuple[Float[Array, "..."], Float[Array, "..."]], Float[Array, "..."]]:
        cm2, cm1 = carry
        cn = (2 * x * (n + a - 1) * cm1 - (n + 2 * a - 2) * cm2) / n
        return (cm1, cn), cn

    _, rest = jax.lax.scan(step, (c0, c1), jnp.arange(2, nmax + 1, dtype=c0.dtype))
    return jnp.concatenate([c0[None], c1[None], rest], axis=0)  # type: ignore[no-any-return]

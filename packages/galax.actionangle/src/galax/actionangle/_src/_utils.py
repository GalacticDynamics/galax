"""Helpers for the action-angle module."""

__all__ = ["chandrupatla", "safe_div", "safe_sqrt", "tang"]

import functools as ft

from collections.abc import Callable

import jax
import jax.numpy as jnp


@ft.partial(jax.jit, inline=True)
def safe_sqrt(q2: jax.Array, /) -> jax.Array:
    """Square root with a finite derivative at 0 (copy of galax.potential's)."""
    return jnp.sqrt(q2 + jnp.finfo(jnp.promote_types(q2.dtype, float)).tiny)  # type: ignore[no-untyped-call]


@ft.partial(jax.jit, inline=True)
def safe_div(a: jax.Array, b: jax.Array, /) -> jax.Array:
    """Divide `a / b`, giving 0 where `b == 0`, with a NaN-free gradient."""
    zero = b == 0
    return jnp.where(zero, 0.0, a / jnp.where(zero, 1.0, b))


@ft.partial(jax.jit, inline=True)
def tang(x: jax.Array, /) -> jax.Array:
    """Return the tangent part of `x`: zero in value, `dx` in derivative."""
    return x - jax.lax.stop_gradient(x)


def _best_and_converged(
    x1: jax.Array, f1: jax.Array, x2: jax.Array, f2: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Endpoint with the smaller ``|f|``, whether the lane is converged, and tl."""
    use1 = jnp.abs(f1) < jnp.abs(f2)
    xm = jnp.where(use1, x1, x2)
    fm = jnp.where(use1, f1, f2)
    width = jnp.abs(x2 - x1)
    tol = 2 * jnp.finfo(width.dtype).eps * jnp.abs(xm)  # type: ignore[no-untyped-call]
    # 2·tol ≥ width ⇔ |b - a| ≤ 4ε|x|
    return xm, (fm == 0) | (2 * tol >= width), safe_div(tol, width)


def chandrupatla(
    f: Callable[[jax.Array], jax.Array],
    a: jax.Array,
    b: jax.Array,
    fa: jax.Array,
    fb: jax.Array,
    x0: jax.Array,
    done0: jax.Array,
    n_iter: int = 16,
) -> jax.Array:
    """Fused, lane-wise Chandrupatla root finder (Scherer's form).

    ``f`` is called once per iteration on the full vector of lanes. ``x0`` is
    each lane's first trial point; ``done0`` lanes return ``x0`` unchanged. A
    lane is frozen once ``f = 0`` or ``|b - a| ≤ 4ε|x|`` and returns the
    endpoint with the smaller ``|f|``. A lane whose bracket is non-finite or
    fails the sign test, or where ``f`` returns NaN, returns NaN. An ``x0``
    not strictly inside the bracket (or NaN) is replaced by the midpoint.
    Non-finite ``f`` values (e.g. an ``±inf`` endpoint) make the
    interpolation fall back to bisection.

    The result carries no gradient (it is wrapped in `jax.lax.stop_gradient`);
    callers reattach tangents.
    """
    valid = jnp.isfinite(a) & jnp.isfinite(b) & (jnp.sign(fa) * jnp.sign(fb) <= 0)
    _, conv0, _ = _best_and_converged(a, fa, b, fb)
    t0 = safe_div(x0 - a, b - a)
    t0 = jnp.where((t0 > 0) & (t0 < 1), t0, 0.5)  # also catches a NaN x0
    init = (a, fa, b, fb, b, fb, t0, done0 | ~valid | conv0, jnp.zeros_like(done0))

    def body(_i: int, s: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        x1, f1, x2, f2, x3, f3, t, done, bad = s
        xt = x1 + t * (x2 - x1)
        f_t = f(xt)
        bad = bad | (~done & jnp.isnan(f_t))  # poisons the lane

        # Shrink the bracket to [xt, x1] or [xt, x2]; x3 is the dropped point.
        same = jnp.sign(f_t) == jnp.sign(f1)
        new = (
            xt,
            f_t,
            jnp.where(same, x2, x1),
            jnp.where(same, f2, f1),
            jnp.where(same, x1, x2),
            jnp.where(same, f1, f2),
        )
        x1, f1, x2, f2, x3, f3 = (
            jnp.where(done, o, n)
            for o, n in zip((x1, f1, x2, f2, x3, f3), new, strict=True)
        )
        _, conv, tl = _best_and_converged(x1, f1, x2, f2)

        # Inverse-quadratic step if (ξ, Φ) is in the validity region, else
        # bisection. Non-finite f's are zeroed so every ratio stays finite.
        fin = jnp.isfinite(f1) & jnp.isfinite(f2) & jnp.isfinite(f3)
        g1, g2, g3 = (jnp.where(fin, fi, 0.0) for fi in (f1, f2, f3))
        xi = safe_div(x1 - x2, x3 - x2)
        phi = safe_div(g1 - g2, g3 - g2)
        xi_in = (xi > 0) & (xi < 1)
        xi_s = jnp.where(xi_in, xi, 0.5)
        interp = fin & xi_in & (1 - jnp.sqrt(1 - xi_s) < phi) & (phi < jnp.sqrt(xi_s))
        t_iq = safe_div(g1, g2 - g1) * safe_div(g3, g2 - g3) + safe_div(
            x3 - x1, x2 - x1
        ) * safe_div(g1, g3 - g1) * safe_div(g2, g3 - g2)
        t_new = jnp.where(interp, t_iq, 0.5)
        t_new = jnp.where(t_new < tl, tl, jnp.where(t_new > 1 - tl, 1 - tl, t_new))

        done = done | conv | bad
        return (x1, f1, x2, f2, x3, f3, jnp.where(done, t, t_new), done, bad)

    x1, f1, x2, f2, *_, bad = jax.lax.fori_loop(0, n_iter, body, init)
    best, _, _ = _best_and_converged(x1, f1, x2, f2)
    root = jnp.where(done0, x0, jnp.where(valid & ~bad, best, jnp.nan))
    return jax.lax.stop_gradient(root)

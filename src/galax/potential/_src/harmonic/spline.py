r"""Radial representation of the multipole coefficient profiles.

Profiles are stored as knot values plus knot derivatives with respect to
:math:`\log r`, not as a fitted spline object, so a `ParameterField` can carry
them as ordinary arrays that may vary with time while keeping C2 accuracy.
Evaluation applies the cubic Hermite basis to those arrays.
"""

__all__: tuple[str, ...] = ()

from jaxtyping import Array, Float

import interpax

import quaxed.numpy as jnp

import galax.potential.custom_types as gt


def radial_grid(n_r: int, r_min: gt.Sz0, r_max: gt.Sz0, /) -> Float[Array, "n_r"]:
    """Return ``n_r`` log-uniformly spaced radii on ``[r_min, r_max]``.

    `jnp.geomspace` pins both endpoints exactly; the boundary fits read them.
    """
    return jnp.geomspace(r_min, r_max, n_r)  # type: ignore[no-any-return]


def fit_log_spline(
    log_r: Float[Array, "n_r"], values: Float[Array, "n_r *rest"], /
) -> Float[Array, "n_r *rest"]:
    r"""Knot derivatives of the cubic spline through ``values``.

    Batched over every trailing axis. With `eval_log_spline` this reproduces
    ``interpax.CubicSpline(..., bc_type="not-a-knot")`` exactly.

    The end condition is load-bearing. A natural one sets the boundary second
    log-derivative to zero by fiat, and a power-law fit to the boundary data
    reads that derivative back out as its exponent -- so the fitted exponent
    would be a property of the end condition rather than of the profile. On a
    Hernquist monopole the boundary derivative is wrong by 1.5e-3 under a
    natural condition against 1.5e-8 under not-a-knot, and that error is a
    floor set by the condition, not a discretization error that ``n_r``
    reduces.
    """
    return interpax.approx_df(log_r, values, "cubic2", 0, bc_type="not-a-knot")  # type: ignore[no-any-return]


def eval_log_spline(
    log_r: Float[Array, "n_r"],
    values: Float[Array, "n_r *rest"],
    derivs: Float[Array, "n_r *rest"],
    log_rq: Float[Array, "*batch"],
    /,
) -> Float[Array, "..."]:
    r"""Evaluate the spline defined by knot values and derivatives.

    Outside the knot range the edge cubic continues: the interval index is
    clamped, the local coordinate is not.

    The Hermite basis is applied directly rather than through
    `interpax.CubicHermiteSpline`, whose object path costs ~13x more per call
    at a single query point, and this runs inside a `diffrax` scan body.
    Knots need not be uniform, hence `searchsorted`.
    """
    idx = jnp.clip(jnp.searchsorted(log_r, log_rq, side="right") - 1, 0, log_r.size - 2)
    h = log_r[idx + 1] - log_r[idx]
    s = (log_rq - log_r[idx]) / h

    s2, s3 = s**2, s**3
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2

    # Broadcast the (*batch,) basis factors against the (*batch, *rest) knots.
    trailing = (1,) * (values.ndim - 1)

    def rs(a: Array) -> Array:
        return a.reshape((*a.shape, *trailing))

    return (
        rs(h00) * values[idx]
        + rs(h10 * h) * derivs[idx]
        + rs(h01) * values[idx + 1]
        + rs(h11 * h) * derivs[idx + 1]
    )

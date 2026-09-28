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


def eval_log_spline_quintic(
    log_r: Float[Array, "n_r"],
    values: Float[Array, "n_r *rest"],
    derivs: Float[Array, "n_r *rest"],
    derivs2: Float[Array, "n_r *rest"],
    log_rq: Float[Array, "*batch"],
    /,
) -> Float[Array, "..."]:
    r"""`eval_log_spline`, but matching the second derivative too.

    The quintic Hermite basis: one degree-5 polynomial per interval agreeing
    with the knot value, first and second log-derivative at both ends, so the
    result is :math:`C^2` and exact for quintics in :math:`\log r` rather
    than cubics.

    This is worth having only because the second derivatives are *free*.
    `solve_poisson_profiles` gets :math:`\Phi_{lm}`, :math:`d\Phi_{lm}/d\log r` and
    :math:`d^2\Phi_{lm}/d\log r^2` from the same two radial integrals -- see
    its docstring -- so nothing extra is computed to feed this. Fitting a
    quintic *spline* instead, from values alone, would cost a solve and give
    no more accuracy than the exact derivatives already carry.

    On a Hernquist monopole over ``[0.05, 20]`` with 128 knots, interpolating
    the exact profile: 5.3e-8 for the cubic basis against 2.8e-13 here.

    Outside the knot range the edge quintic continues, as in `eval_log_spline`
    -- the interval index is clamped, the local coordinate is not. A quintic
    diverges faster than a cubic once past the end, which is why
    `eval_log_spline_asympt` clamps the query before calling this and uses the
    fitted power law beyond the grid instead.
    """
    idx = jnp.clip(jnp.searchsorted(log_r, log_rq, side="right") - 1, 0, log_r.size - 2)
    h = log_r[idx + 1] - log_r[idx]
    s = (log_rq - log_r[idx]) / h

    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s
    h00 = 1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5
    h10 = s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5
    h20 = 0.5 * s2 - 1.5 * s3 + 1.5 * s4 - 0.5 * s5
    h01 = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
    h11 = -4.0 * s3 + 7.0 * s4 - 3.0 * s5
    h21 = 0.5 * s3 - s4 + 0.5 * s5

    trailing = (1,) * (values.ndim - 1)

    def rs(a: Array) -> Array:
        return a.reshape((*a.shape, *trailing))

    h2 = h * h
    return (
        rs(h00) * values[idx]
        + rs(h10 * h) * derivs[idx]
        + rs(h20 * h2) * derivs2[idx]
        + rs(h01) * values[idx + 1]
        + rs(h11 * h) * derivs[idx + 1]
        + rs(h21 * h2) * derivs2[idx + 1]
    )

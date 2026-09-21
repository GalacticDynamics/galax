r"""Radial representation of the multipole coefficient profiles.

Radial profiles are stored as knot values plus knot derivatives with respect
to :math:`\log r`, rather than as a fitted spline object. The knot derivatives
come from a cubic-spline solve at build time, so the coefficients stay
ordinary arrays that a `ParameterField` can carry and that may vary with time,
while retaining C2 accuracy. Evaluation then applies the cubic Hermite basis
directly to those arrays.

This module's single job is that representation: laying out the knots, fitting
the derivatives, and evaluating. It carries no harmonic content and is tested
as the generic utility it is.
"""

__all__: tuple[str, ...] = ()

from jaxtyping import Array, Float

import interpax

import quaxed.numpy as jnp

import galax.potential.custom_types as gt


def radial_grid(n_r: int, r_min: gt.Sz0, r_max: gt.Sz0, /) -> Float[Array, "n_r"]:
    """Return ``n_r`` log-uniformly spaced radii on ``[r_min, r_max]``."""
    return jnp.exp(jnp.linspace(jnp.log(r_min), jnp.log(r_max), n_r))  # type: ignore[no-any-return]


def fit_log_spline(
    log_r: Float[Array, "n_r"], values: Float[Array, "n_r *rest"], /
) -> Float[Array, "n_r *rest"]:
    r"""Knot derivatives of the cubic spline through ``values``.

    Batched over every trailing axis. Paired with `eval_log_spline` this
    reproduces ``interpax.CubicSpline(..., bc_type="not-a-knot")`` exactly.

    The end condition is load-bearing, and not merely for the last knot.
    `asymptotic_coeffs` fits each tail's exponent from the boundary data, and
    expanding its residual in the knot spacing shows the fitted slope is, to
    leading order, a readout of the boundary *second* log-derivative:

    .. math::

        s = (\Phi'' - P_1 v^2) / K - v ,
        \qquad \Phi'' \equiv d^2\Phi/d(\ln r)^2 .

    A natural end condition sets exactly that quantity to zero by fiat, so
    the tail exponent converges to a number fixed by the boundary condition
    rather than by the density. Measured on a Hernquist monopole, natural
    against not-a-knot:

    ============================ ========== ==========
    quantity (n_r = 1024)        natural    not-a-knot
    ============================ ========== ==========
    boundary derivative rel.err  1.5e-3     1.5e-8
    outward tail at 2 r_max      1.7e-2     1.3e-4
    inward tail at r_min / 2     4.5e-3     1.3e-4
    ============================ ========== ==========

    Note the natural figures barely move with ``n_r`` -- 1.8e-2 at 256
    against 1.7e-2 at 1024 -- because they are a floor set by the end
    condition, not a discretization error.
    """
    return interpax.approx_df(log_r, values, "cubic2", 0, bc_type="not-a-knot")  # type: ignore[no-any-return]


def eval_log_spline(
    log_r: Float[Array, "n_r"],
    values: Float[Array, "n_r *rest"],
    derivs: Float[Array, "n_r *rest"],
    log_rq: Float[Array, "*batch"],
    /,
) -> Array:
    r"""Evaluate the spline defined by knot values and derivatives.

    Outside ``[log_r[0], log_r[-1]]`` the edge cubic is extrapolated: the
    interval index is clamped while the local coordinate is not, so the edge
    cubic simply continues. `eval_log_spline_asympt` is the guarded form that
    replaces that continuation with a fitted power law.

    The cubic Hermite basis is applied directly rather than by constructing an
    `interpax.CubicHermiteSpline`. That constructor materializes a
    ``(4, n_r - 1, *rest)`` power-basis coefficient array from the knot data,
    which is loop-invariant -- but this function is called from inside a
    `diffrax` solver scan body, so the construction was re-executed on every
    integration step (~53 us of fixed cost per call, dominating orbit
    integration while staying invisible in a single large batch).

    ``log_r`` is not assumed uniform: `r_knots` is a public parameter and may
    carry arbitrary knots, so the interval is found by `searchsorted`.
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

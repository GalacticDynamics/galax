r"""Radial representation and the expansion build pipeline.

Radial profiles are stored as knot values plus knot derivatives with respect
to :math:`\log r`, rather than as a fitted spline object. The knot derivatives
come from a natural-cubic solve at build time, so the coefficients stay
ordinary arrays that a `ParameterField` can carry and that may vary with time,
while retaining C2 accuracy. Evaluation then applies the cubic Hermite basis
directly to those arrays.

Inner cusp subtraction
----------------------
For a steep inner slope (:math:`\rho_{lm} \sim A r^\alpha` near the origin),
splining :math:`\rho_{lm}` directly is ill-conditioned near ``r_min`` and
extrapolates badly inward. The power-law background is estimated from the
innermost grid points and subtracted before fitting; the residual splines
cleanly and the background is added back analytically at evaluation.

The background is written :math:`\rho_{lm}(r_0) (r/r_0)^\alpha` rather than
``bfeax``'s :math:`A r^\alpha`: identical values, but the stored amplitude is
a plain mass density instead of carrying units of
density / length\ :sup:`alpha`, which no single dimension can express.
"""

__all__: tuple[str, ...] = ()

import functools as ft

from collections.abc import Callable
from jaxtyping import Array, Float

import interpax
import jax

import quaxed.numpy as jnp

import galax.potential.custom_types as gt
from .poisson import solve_poisson_lm
from .project import project_density

_LOG_FLOOR: float = 1e-300
"""Floor inside ``log|rho|``, matching ``bfeax`` exactly (bit-fidelity)."""

_CUSP_TOL: float = 1e-6
"""Relative gate on the cusp background. NB: ``bfeax`` uses 1e-6 here against
a *global* scale, distinct from the Poisson solve's 1e-8 against a *per-mode*
scale. Both are reproduced as written."""


def radial_grid(n_r: int, r_min: gt.Sz0, r_max: gt.Sz0, /) -> Float[Array, "n_r"]:
    """Return ``n_r`` log-uniformly spaced radii on ``[r_min, r_max]``."""
    return jnp.exp(jnp.linspace(jnp.log(r_min), jnp.log(r_max), n_r))  # type: ignore[no-any-return]


def fit_log_spline(
    log_r: Float[Array, "n_r"], values: Float[Array, "n_r *rest"], /
) -> Float[Array, "n_r *rest"]:
    r"""Knot derivatives of the natural cubic spline through ``values``.

    Batched over every trailing axis. Paired with `eval_log_spline` this
    reproduces ``interpax.CubicSpline(..., bc_type="natural")`` exactly.
    """
    return interpax.approx_df(log_r, values, "cubic2", 0, bc_type="natural")  # type: ignore[no-any-return]


def eval_log_spline(
    log_r: Float[Array, "n_r"],
    values: Float[Array, "n_r *rest"],
    derivs: Float[Array, "n_r *rest"],
    log_rq: Float[Array, "*batch"],
    /,
) -> Array:
    r"""Evaluate the spline defined by knot values and derivatives.

    Outside ``[log_r[0], log_r[-1]]`` the edge cubic is extrapolated, matching
    ``bfeax``'s clamped-index evaluation: the interval index is clamped while
    the local coordinate is not, so the edge cubic simply continues.

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


@ft.partial(jax.jit)
def subtract_inner_cusp(
    r_knots: Float[Array, "n_r"], rho_lm: Float[Array, "n_r n_modes"], /
) -> tuple[
    Float[Array, "n_r n_modes"], Float[Array, "n_modes"], Float[Array, "n_modes"]
]:
    r"""Split :math:`\rho_{lm}` into a splineable residual and a power law.

    Returns ``(residual, alpha, amplitude)`` with

    .. math::

        \rho_{lm}(r) = \mathrm{residual}(r)
                     + \mathrm{amplitude} \left(\frac{r}{r_0}\right)^{\alpha}

    where :math:`r_0` is the innermost knot, so ``amplitude`` is just
    :math:`\rho_{lm}(r_0)`. The slope is a log-log finite difference over the
    innermost three knots. Modes negligible at ``r_min`` relative to the
    global coefficient scale, and modes that change sign within that window,
    get a zero background rather than a slope fitted to numerical noise or to
    a spurious crossing; the residual then carries the mode in full.
    """
    log_ratio = jnp.log(r_knots / r_knots[0])
    global_scale = jnp.max(jnp.abs(rho_lm))

    log_inner = jnp.log(jnp.abs(rho_lm[:3, :]) + _LOG_FLOOR)
    alpha = jnp.mean(
        jnp.diff(log_inner, axis=0) / jnp.diff(jnp.log(r_knots[:3]))[:, None],
        axis=0,
    )
    amplitude = rho_lm[0, :]

    # `alpha` is a slope of log|rho_lm|, so a zero crossing inside the
    # three-knot window turns a decaying mode into a large *positive* fitted
    # slope and the background then diverges outward (alpha ~ +30 observed,
    # background ~1e50, catastrophic cancellation in residual + background).
    # A magnitude gate on rho_lm[0] alone cannot see this, so require the
    # window not to change sign before accepting any background at all.
    same_sign = jnp.all(
        jnp.sign(rho_lm[:3, :]) == jnp.sign(rho_lm[0, :])[None, :], axis=0
    )
    valid = (jnp.abs(rho_lm[0, :]) > _CUSP_TOL * global_scale) & same_sign
    # Second line of defence: a physical inner logarithmic slope sits well
    # inside +/-3 (r^-2 isothermal and r^-1 NFW cusps at one end, an analytic
    # core at the other), so clip rather than trust a noisy three-point fit.
    alpha = jnp.clip(alpha, -3.0, 3.0)
    alpha = jnp.where(valid, alpha, 0.0)
    amplitude = jnp.where(valid, amplitude, 0.0)

    background = amplitude[None, :] * jnp.exp(alpha[None, :] * log_ratio[:, None])
    return rho_lm - background, alpha, amplitude


@ft.partial(jax.jit, static_argnums=(0, 2, 3, 4, 5))
def build_expansion(
    rho_fn: Callable[[gt.BtSz3, gt.BBtSz0], Float[Array, "..."]],
    r_knots: Float[Array, "n_r"],
    l_max: int,
    keys: tuple[tuple[int, int], ...],
    n_theta: int,
    n_phi: int,
    t: gt.BBtSz0,
    G: gt.Sz0,
    /,
) -> dict[str, Array]:
    r"""Project, solve, and fit — the whole build in one jitted call.

    Returns the six coefficient arrays that `MultipoleProfilePotential`
    stores as parameters.
    """
    log_r = jnp.log(r_knots)
    l_per_mode = jnp.asarray([float(l) for l, _ in keys])

    rho_lm = project_density(rho_fn, r_knots, l_max, keys, n_theta, n_phi, t)
    phi_lm = solve_poisson_lm(r_knots, rho_lm, l_per_mode, G)
    residual, alpha, amplitude = subtract_inner_cusp(r_knots, rho_lm)

    return {
        "phi_lm": phi_lm,
        "dphi_lm": fit_log_spline(log_r, phi_lm),
        "rho_residual_lm": residual,
        "drho_residual_lm": fit_log_spline(log_r, residual),
        "rho_alpha": alpha,
        "rho_amplitude": amplitude,
    }

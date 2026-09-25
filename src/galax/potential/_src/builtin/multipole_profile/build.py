r"""The multipole profile build pipeline.

This module's single job is to turn a density callable into the coefficient
arrays `MultipoleProfilePotential` stores: project it onto harmonics
(`project`), solve the radial Poisson equation (`poisson`), and fit the
resulting profiles (`spline`).

Inner cusp subtraction
----------------------
For a steep inner slope (:math:`\rho_{lm} \sim A r^\alpha` near the origin),
splining :math:`\rho_{lm}` directly is ill-conditioned near ``r_min`` and
extrapolates badly inward. The power-law background is estimated from the
innermost grid points and subtracted before fitting; the residual splines
cleanly and the background is added back analytically at evaluation.

The background is written :math:`\rho_{lm}(r_0) (r/r_0)^\alpha` rather than
the equivalent :math:`A r^\alpha`: identical values, but the stored amplitude
is a plain mass density instead of carrying units of
density / length\ :sup:`alpha`, which no single dimension can express and so
no `ParameterField` could hold.
"""

__all__: tuple[str, ...] = ()

import functools as ft

from collections.abc import Callable
from jaxtyping import Array, Float

import jax
import numpy as np

import quaxed.numpy as jnp

import galax.potential.custom_types as gt
from galax.potential._src.harmonic import (
    asymptotic_coeffs,
    fit_log_spline,
    harmonic_coeffs,
    solve_poisson_lm,
)


def _log_floor(x: Float[Array, "..."], /) -> float:
    r"""Smallest coefficient magnitude treated as non-zero inside ``log``.

    Sized from the working dtype: a float64 constant such as ``1e-300``
    underflows to *exactly zero* in float32 -- which is what a caller gets,
    since `galax` does not enable x64 on import -- so the floor stops
    flooring and an identically-zero mode takes ``log(0) = -inf``.

    TODO: share this with the identical helper in `harmonic.poisson` once
    this branch rebases onto a `main` that carries it (#870).
    """
    return 16.0 * float(np.finfo(x.dtype).tiny)


_CUSP_TOL: float = 1e-6
"""Relative gate on the cusp background, against a *global* scale.

Deliberately distinct from the Poisson solve's ``_ACTIVE_TOL``, which is 1e-8
against a *per-mode* scale: a mode negligible next to the largest mode in the
expansion need not be negligible next to itself.
"""


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

    log_inner = jnp.log(jnp.abs(rho_lm[:3, :]) + _log_floor(rho_lm))
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

    Returns the coefficient arrays `MultipoleProfilePotential` stores as
    parameters. `expansion`'s module docstring lists the keys; keeping the
    count out of this sentence keeps the two from drifting apart.
    """
    log_r = jnp.log(r_knots)
    l_per_mode = jnp.asarray([float(l) for l, _ in keys])

    rho_lm = harmonic_coeffs(rho_fn, r_knots, l_max, keys, n_theta, n_phi, t)
    phi_lm = solve_poisson_lm(r_knots, rho_lm, l_per_mode, G)
    dphi_lm = fit_log_spline(log_r, phi_lm)
    residual, alpha, amplitude = subtract_inner_cusp(r_knots, rho_lm)

    # `(v, s, B)` per side: `v`/`s` are exponents and `B` scales a potential,
    # so they are split into a dimensionless and a specific-energy field
    # rather than stored as one array of mixed dimensions. Split at a fixed
    # index rather than in half -- `asymptotic_coeffs` carries a fourth `Q`
    # row only under `cored_monopole`, which this build does not ask for.
    coefs = asymptotic_coeffs(log_r, phi_lm, dphi_lm, l_per_mode)
    powers, scales = coefs[:, :2], coefs[:, 2:]

    return {
        "phi_lm": phi_lm,
        "dphi_lm": dphi_lm,
        "phi_asympt_powers": powers,
        "phi_asympt_scales": scales,
        "rho_residual_lm": residual,
        "drho_residual_lm": fit_log_spline(log_r, residual),
        "rho_alpha": alpha,
        "rho_amplitude": amplitude,
    }

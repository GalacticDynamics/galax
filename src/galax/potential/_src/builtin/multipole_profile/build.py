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
)
from galax.potential._src.harmonic.poisson import (
    gl_log_nodes,
    solve_poisson_profiles,
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


_GL_NODES: int = 4
r"""Gauss-Legendre nodes per radial interval in the Poisson solve.

Four is where the rule stops paying. Integrating a Hernquist monopole over
``[0.05, 20]`` on 128 intervals, against the closed form at the knots:

======= ==========
nodes    max rel
======= ==========
1        4.3e-5
2        1.3e-9
3        2.2e-14
4        3.5e-16
8        3.5e-16
======= ==========

Four reaches round-off and more nodes do not improve it, so the cost -- one
density evaluation per node per interval, paid once at build time -- buys
nothing beyond this.
"""

_PAD_MULTIPLE: int = 2
r"""Knots added beyond each end of the grid, as ``n_r * _PAD_MULTIPLE``.

The Poisson solve models the mass outside ``[r_min, r_max]`` as a single
power law fitted at the boundary. That model cannot represent a profile with
curvature there, and -- unlike the interior quadrature -- its error does not
improve with ``n_r``. Unpadded, refining from ``n_r = 128`` to ``2048`` moved
the Hernquist monopole error only 1.2e-3 to 6.8e-4: the floor, not the
resolution, was the answer.

Fitting the slope better does not help. The local three-point slope the solve
already uses beats both the true asymptotic slope and a wider two-point
baseline -- the power-law *form* is the limit. The fix is to put the boundary
where the model is not asked to carry the answer: solve on a padded grid and
keep only the range the caller asked for.

How far to pad is set by what it has to beat, so it has moved twice as the
rest of the solve improved. Against the closed form over ``[0.06, 18]``:

========== ============ ============ ============
padding     trapezoid    +Hermite     +GL, quintic
========== ============ ============ ============
50%         2.9e-5       1.6e-7       1.6e-7
100%        2.9e-5       3.6e-11      1.9e-11
200%        --           3.6e-11      4.1e-15
400%        --           --           4.1e-15
========== ============ ============ ============

Each column is at the resolution where that configuration stops improving
(``n_r = 512``, 256 for the last). 50% was the plateau while the interior
rule was the trapezoid, because the quadrature swamped the tail; raising the
rule to fourth order made 100% worth paying for, and sampling the density at
Gauss-Legendre nodes moved it again to 200%. 300% and 600% measure the same
as 200%, so this is the plateau and not another step along it.

The cost is one-off, at build time: the solve grid is ``5 * n_r`` knots and
each carries `_GL_NODES` density evaluations.
"""


def _pad_grid(r_knots: Float[Array, "n_r"], /) -> tuple[Float[Array, "n_pad"], int]:
    """Extend ``r_knots`` at both ends, continuing its own spacing.

    Returns the padded grid and the index at which the original knots start,
    so the solve's output can be sliced back without interpolating. The count
    is taken from the shape, which is static under `jax.jit`; the spacing is
    read from the values, so a grid that is not log-uniform still continues
    smoothly from each end.
    """
    n_r = r_knots.shape[0]
    # Four knots, matching `MultipoleProfilePotential.from_density`: the
    # boundary slopes are fitted over three. This has to be checked *before*
    # padding, because padding defeats the `n_r >= 3` guard inside
    # `solve_poisson_lm` -- a single knot pads to three and sails through it.
    # It also has to be checked at all: `r_knots[1]` on a one-knot grid is out
    # of bounds, and JAX clamps rather than raising, so the ratio comes out as
    # 1 and every padded knot lands on top of the original.
    if n_r < 4:
        msg = (
            f"r_knots must have at least 4 entries (got {n_r}); the boundary "
            "slopes are fitted over three"
        )
        raise ValueError(msg)
    n_pad = max(1, n_r * _PAD_MULTIPLE)
    ratio_lo = r_knots[1] / r_knots[0]
    ratio_hi = r_knots[-1] / r_knots[-2]
    lo = r_knots[0] * ratio_lo ** jnp.arange(-n_pad, 0)
    hi = r_knots[-1] * ratio_hi ** jnp.arange(1, n_pad + 1)
    return jnp.concat([lo, r_knots, hi]), n_pad


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

    # Solve on a padded grid so the boundary tail models sit outside the range
    # the caller asked for, then keep only that range. See `_PAD_MULTIPLE`.
    r_solve, lo = _pad_grid(r_knots)
    n_r = r_knots.shape[0]
    rho_solve = harmonic_coeffs(rho_fn, r_solve, l_max, keys, n_theta, n_phi, t)
    rho_lm = rho_solve[lo : lo + n_r]

    # Sample the density *inside* each interval as well, at Gauss-Legendre
    # nodes, so the radial integrals are quadrature rather than interpolation.
    # This is the one thing only the builder can do: `solve_poisson_lm` is
    # handed an array and cannot ask for more of it, so on knots alone it is
    # capped by how well a rule reconstructs rho between them. `rho_fn` can be
    # called anywhere, and four nodes per interval is enough to integrate a
    # smooth profile to round-off.
    log_gl, _ = gl_log_nodes(jnp.log(r_solve), _GL_NODES)
    rho_gl = harmonic_coeffs(
        rho_fn, jnp.exp(log_gl).reshape(-1), l_max, keys, n_theta, n_phi, t
    ).reshape(log_gl.shape[0], _GL_NODES, -1)

    # The solve returns the first and second log-derivatives alongside the
    # profile. Both fall out of the two radial integrals it already formed,
    # so they cost nothing and are exact where a spline fit of `phi_lm` is
    # only as good as the fit -- which is what lets `expansion_potential`
    # interpolate with the quintic basis.
    phi_all, dphi_all, d2phi_all = solve_poisson_profiles(
        r_solve, rho_solve, l_per_mode, G, rho_gl
    )
    sl = slice(lo, lo + n_r)
    phi_lm, dphi_lm, d2phi_lm = phi_all[sl], dphi_all[sl], d2phi_all[sl]
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
        "d2phi_lm": d2phi_lm,
        "phi_asympt_powers": powers,
        "phi_asympt_scales": scales,
        "rho_residual_lm": residual,
        "drho_residual_lm": fit_log_spline(log_r, residual),
        "rho_alpha": alpha,
        "rho_amplitude": amplitude,
    }

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


_LN_HUGE_FRAC: float = 0.985
"""Fraction of the dtype's overflow exponent the padded grid may use."""

_GL_NODES: int = 5
r"""Gauss-Legendre nodes per radial interval in the Poisson solve.

Four reaches round-off on the *monopole*, which is what an earlier table here
measured and why this was 4. It is not enough at higher :math:`l`: the
integrand carries :math:`x^{l+3}`, so the rule has a higher-degree function
to integrate as :math:`l` grows, and the node count has to follow.

Against the closed form for :math:`\rho = r^{-1.5}` at ``n_r = 128``, padded
(the outer integral needs :math:`l > 0.5` to converge, so the monopole is not
in this table):

======= ========= ========= ========= =========
nodes    l=2       l=4       l=8       l=12
======= ========= ========= ========= =========
3        9.6e-11   1.3e-09   2.0e-08   9.0e-08
4        1.2e-14   3.8e-13   1.6e-11   1.3e-10
5        4.4e-16   5.6e-16   8.8e-15   1.4e-13
6        4.4e-16   6.7e-16   1.6e-15   2.4e-15
======= ========= ========= ========= =========

Five is where it stops paying for the :math:`l` a caller is likely to ask
for: it is at round-off through :math:`l = 4` and within a factor of ten of
it at :math:`l = 8`, where four leaves 1.6e-11. Six buys another two orders
at :math:`l = 12` alone, which is not worth 20% more density evaluations.

On a Hernquist monopole the whole build measures 2.7e-13 at every one of
these -- the boundary tail, not the quadrature, is what binds there -- so
this constant is tuned on the high-:math:`l` columns.

Cost is one density evaluation per node per interval, paid once at build
time, and `_PAD_KNOTS` caps the number of intervals it applies to.
"""

_PAD_KNOTS: int = 128
r"""Cap on padded knots per side.

The pad only has to carry a smooth power law out to `_PAD_MULTIPLE` spans,
so it does not need the caller's resolution. Capping it decouples build cost
from `n_r`: at ``n_r = 512`` this is 128 knots a side rather than 1024, and
the padded grid goes from 2560 knots to 768. Measured at ``l_max = 8``,
``n_r = 512``, the whole build drops from 60.7 ms to 18.1 ms -- 3.4x -- and
the Hernquist monopole at that resolution is *better*, 8.7e-16 against
5.9e-16, since the coarser pad also shortens the sums it accumulates.

The cap and `_GL_NODES` were raised together: coarsening the pad puts more
weight on each interval's rule, which is what five nodes pay for.
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

The cost is one-off, at build time. The solve grid is
``n_r + 2 * min(n_r * _PAD_MULTIPLE, _PAD_KNOTS)`` knots, each carrying
`_GL_NODES` density evaluations -- so the padding stops growing with ``n_r``
once the cap binds, and the grid tends to ``n_r + 256`` rather than
``5 * n_r``:

======= ========== ==========
``n_r``  padded     uncapped
======= ========== ==========
64       320        320
256      512        1280
512      768        2560
2048     2304       10240
======= ========== ==========
"""


def _check_enough_knots(n_r: int, /) -> None:
    """Reject a grid too small for the three-knot boundary fits.

    Four knots, matching `MultipoleProfilePotential.from_density`: the
    boundary slopes are fitted over three. This has to be checked *before*
    padding, because padding defeats the ``n_r >= 3`` guard inside the radial
    solve -- a single knot pads to three and sails through it.

    It also has to be checked at all: ``r_knots[1]`` on a one-knot grid is out
    of bounds, and JAX clamps rather than raising, so the ratio comes out as 1
    and every padded knot lands on top of the original.
    """
    if n_r < 4:
        msg = (
            f"r_knots must have at least 4 entries (got {n_r}); the boundary "
            "slopes are fitted over three"
        )
        raise ValueError(msg)


def _requested_reach(log_r: Float[Array, "n_r"], /) -> Float[Array, ""]:
    """Pad reach asked for by `_PAD_MULTIPLE`, capped by the solver's own range.

    The largest power the solve forms is :math:`x^2` about the padded grid's
    log-midpoint, so the padded span must fit the dtype's overflow exponent;
    everything carrying an :math:`l` is held in scaled form (see
    `harmonic.poisson._scaled_prefix`) and is bounded by this.

    A caller asking for ``[1e-6, 1e6]`` already spans 27.6 e-folds; two spans
    either side would be 138, past float32's 87.4, and the build came back
    `nan`. Capping degrades gracefully: the pad stops growing, so such a
    caller gets less tail accuracy rather than no answer. In float64 the
    budget is 699 and nothing physical comes close.

    This bounds `galax`'s *own* arithmetic only, and is deliberately the only
    thing that sets the reach. Whether the caller's density survives being
    evaluated that far out is a separate question, answered by zeroing what
    cannot be represented and anchoring the radial solve's inner tail above
    it -- see `solve_poisson_profiles`. Shortening the pad instead was tried
    and measured worse: it costs every caller tail accuracy to avoid an
    overflow the tail already handles.
    """
    span = log_r[-1] - log_r[0]
    budget = _LN_HUGE_FRAC * float(np.log(np.finfo(log_r.dtype).max))
    reach: Float[Array, ""] = jnp.minimum(
        _PAD_MULTIPLE * span, 0.5 * jnp.maximum(budget - span, 0.0)
    )
    return reach


def _pad_grid(r_knots: Float[Array, "n_r"], /) -> tuple[Float[Array, "n_pad"], int]:
    """Extend ``r_knots`` at both ends, continuing its own spacing.

    Returns the padded grid and the index at which the original knots start,
    so the solve's output can be sliced back without interpolating. The count
    is taken from the shape, which is static under `jax.jit`; the spacing is
    read from the values, so a grid that is not log-uniform still continues
    smoothly from each end.

    How far it reaches is `_requested_reach`, which bounds the solve's own
    arithmetic. It deliberately does *not* consult the caller's density:
    where that density is unrepresentable the samples are zeroed and the
    radial solve anchors its inner tail above them, which costs far less
    than shortening the pad for everyone. See `solve_poisson_profiles`.
    """
    n_r = r_knots.shape[0]
    _check_enough_knots(n_r)
    # Pad by log *reach*, not by knot count, and take the step from the
    # grid's whole span rather than its end intervals.
    #
    # Extending at the end interval's own ratio assumed the caller's grid is
    # log-uniform. On a linearly spaced grid it is not: `ratio_lo` is large
    # and `ratio_hi` is ~1, so the two ends pad by wildly different reaches
    # and the inner one underflows -- `linspace(0.05, 20, 128)` reached
    # `5e-160` in float64 and *exactly zero* in float32, where `log(0)` is
    # `-inf` and every output is `nan`.
    #
    # Reach is what the padding is for (see `_PAD_MULTIPLE`), and the pad
    # does not need the caller's resolution to deliver it: rho out there is a
    # smooth power law, which is why `_PAD_KNOTS` can be a cap rather than a
    # multiple of `n_r`. At n_r=512 that is 128 knots a side instead of 1024,
    # a 3.4x cheaper build that matches the old one to round-off.
    log_r = jnp.log(r_knots)
    reach = _requested_reach(log_r)
    n_pad = min(n_r * _PAD_MULTIPLE, _PAD_KNOTS)
    step = reach / n_pad
    lo = jnp.exp(log_r[0] - step * jnp.arange(n_pad, 0, -1))
    hi = jnp.exp(log_r[-1] + step * jnp.arange(1, n_pad + 1))
    return jnp.concat([lo, r_knots, hi]), n_pad


def _drop_nonfinite(rho: Float[Array, "..."], /) -> Float[Array, "..."]:
    r"""Zero any density sample the working dtype could not represent.

    Padding evaluates the caller's ``rho_fn`` outside the bracket it was
    asked about, so an inner cusp :math:`\rho \sim r^{-\gamma}` is sampled
    where the *density* can overflow, once
    :math:`\gamma \times \mathrm{reach}` clears the dtype's exponent.

    Zeroing is safe because it is no longer the whole story: the radial solve
    anchors its inner tail at the first knot above the deepest zeroed sample,
    so the band those samples should have carried is covered analytically
    rather than lost. Without that, zeroing `rho[0]` also switched off the
    tail and cost 11% of the monopole; with it, ~5e-06.

    Without this, one unrepresentable sample took out the whole build. The
    projection turns ``inf`` into ``inf`` at :math:`l = 0` and ``nan`` above
    it, and `fit_log_spline` solves one system per mode across the entire
    padded grid, so a single bad row poisons every radius and every mode --
    silently, since `jit` raises nothing. A Dehnen :math:`\gamma = 2.5`
    cusp over ``[1e-4, 1e4]`` at ``l_max = 8`` produced 41472 non-finite
    values out of 41472 in float32, and none in float64.

    Zeroing is not free, which is why it is no longer the primary guard: a
    zeroed ``rho[0]`` also switches off the solve's analytic inner tail,
    which is gated on that sample being non-negligible, so the loss is the
    dropped band *plus* everything inside it. At :math:`\gamma = 2.9` that
    was 11% of the monopole. It does not touch the requested bracket, where
    every sample is representable by construction.
    """
    out: Float[Array, "..."] = jnp.where(jnp.isfinite(rho), rho, 0.0)
    return out


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
    _check_enough_knots(r_knots.shape[0])
    log_r = jnp.log(r_knots)
    l_per_mode = jnp.asarray([float(l) for l, _ in keys])

    # Solve on a padded grid so the boundary tail models sit outside the range
    # the caller asked for, then keep only that range. See `_PAD_MULTIPLE`.
    r_solve, lo = _pad_grid(r_knots)

    n_r = r_knots.shape[0]
    rho_raw = harmonic_coeffs(rho_fn, r_solve, l_max, keys, n_theta, n_phi, t)
    # Where the caller's density could not be represented, per mode. The
    # samples are zeroed below, but the solve needs to know *which* ones: its
    # inner tail is anchored at the first knot above the deepest bad sample,
    # so the band those samples should have carried is covered analytically
    # instead of silently lost. See `solve_poisson_profiles`.
    #
    # The deepest bad index, not the first good one: an unrepresentable band
    # can sit mid-pad rather than at its inner edge -- a density need not be
    # monotonic in log r -- and everything below such a band is suspect even
    # where it happens to sample finite.
    bad = ~jnp.isfinite(rho_raw)
    idx = jnp.arange(r_solve.shape[0])[:, None]
    first_ok = jnp.max(jnp.where(bad, idx + 1, 0), axis=0)
    # Never discard the caller's own range: if their density is not finite
    # inside the bracket they asked for, there is nothing the pad can do and
    # anchoring there would throw away real interior data.
    first_ok = jnp.minimum(first_ok, lo)

    rho_solve = _drop_nonfinite(rho_raw)
    rho_lm = rho_solve[lo : lo + n_r]

    # Sample the density *inside* each interval as well, at Gauss-Legendre
    # nodes, so the radial integrals are quadrature rather than interpolation.
    # This is the one thing only the builder can do: `solve_poisson_profiles`
    # is handed an array and cannot ask for more of it, so on knots alone it
    # is capped by how well a rule reconstructs rho between them. `rho_fn` can
    # be called anywhere; `_GL_NODES` nodes per interval is what that costs,
    # and its docstring is where the count is justified.
    log_gl, _ = gl_log_nodes(jnp.log(r_solve), _GL_NODES)
    rho_gl = _drop_nonfinite(
        harmonic_coeffs(
            rho_fn, jnp.exp(log_gl).reshape(-1), l_max, keys, n_theta, n_phi, t
        ).reshape(log_gl.shape[0], _GL_NODES, -1)
    )

    # The solve returns the first and second log-derivatives alongside the
    # profile. Both fall out of the two radial integrals it already formed,
    # so they cost nothing and are exact where a spline fit of `phi_lm` is
    # only as good as the fit -- which is what lets `expansion_potential`
    # interpolate with the quintic basis.
    phi_all, dphi_all, d2phi_all = solve_poisson_profiles(
        r_solve, rho_solve, l_per_mode, G, rho_gl, first_ok
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

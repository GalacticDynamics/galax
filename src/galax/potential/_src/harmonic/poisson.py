r"""Radial Poisson solve for each spherical harmonic mode.

.. math::

    \Phi_{lm}(r) = -\frac{4\pi G}{2l+1} \left[
        r^{-(l+1)} \int_0^r \rho_{lm}(r') r'^{l+2} dr'
        + r^l \int_r^\infty \rho_{lm}(r') r'^{1-l} dr'
    \right]

Boundary tails
--------------
Truncating the grid at ``r_min`` and ``r_max`` biases both integrals. The
power-law slope of :math:`\rho_{lm}` is estimated at each boundary from three
grid points and an analytic tail appended. Both tails are guarded for
convergence, but the two boundary-value gates deliberately differ: the inner
one requires :math:`|\rho_{lm}(r_\min)|` to exceed ``_active_tol`` times the
per-mode scale, while the outer one requires only
:math:`\rho_{lm}(r_\max) \neq 0`. See "Outer-tail sign" for why the inner
threshold is not reused at the outer boundary.

Choosing ``n_r``
----------------
The interior rule is cubic-Hermite quadrature in :math:`\log r`: the
trapezoid plus the endpoint-slope correction that makes it exact for cubics,
taking its slopes from the same not-a-knot fit the profiles are stored with.
Error is :math:`O(h^4)`. It still samples the :math:`r^{l+2}` factor rather
than integrating it, so the error grows with :math:`l` while :math:`\rho`
stays fixed. Measured against the closed form for :math:`\rho = r^{-1.5}`,
interior knots only:

======= ========= ========= =========
l       n_r=256   n_r=512   n_r=1024
======= ========= ========= =========
2       5.7e-7    3.6e-8    2.2e-9
4       5.2e-6    3.3e-7    2.1e-8
8       5.9e-5    3.9e-6    2.5e-7
12      2.5e-4    1.8e-5    1.1e-6
======= ========= ========= =========

So ``n_r`` should still rise with ``l_max``, but far more slowly than under a
second-order rule: one halving of the step buys a factor of sixteen, which
covers roughly eight levels of :math:`l`.

In float32 -- which is `galax`'s default, while the test suite forces x64 --
the rule reaches its round-off floor of a few times ``eps``, about 3e-7 on
the monopole above, at ``n_r`` near 256; refining past that buys nothing.
The second-order rule it replaces never got close enough to the floor for it
to matter, so this ceiling is new. At ``n_r=256`` the same measurement is
1.1e-3 under the trapezoid against 3.2e-7 here.

Two alternatives were measured and rejected. Integrating :math:`r^{l+2}`
exactly while interpolating only :math:`\rho` removes the :math:`l`
dependence but is 4.5x *less* accurate on a Hernquist monopole, because real
:math:`\rho_{lm}` profiles curve in log-log while that rule assumes they do
not. The plain trapezoid in :math:`r` is :math:`O(h^2)`: on a Hernquist
monopole over a wide bracket it gives 6.8e-5 at ``n_r=1024`` where this rule
gives 1.5e-10.

Sampling :math:`\rho_{lm}` *inside* each interval -- Gauss-Legendre nodes
rather than knots alone -- reaches 3.5e-16 with four nodes on 128 intervals,
but needs the density evaluated off the knot grid, which only the caller can
do.

Outer-tail sign
---------------
The outer-tail gate tests :math:`\rho_{lm}(r_\max) \neq 0`, not
:math:`\rho_{lm}(r_\max) > 0`. The sign carries no information about whether
the tail converges -- that is what the exponent decides -- so gating on it
would silently drop the correction for every mode whose outermost coefficient
happens to be negative, which is routine for :math:`l \ge 1`. For a triaxial
NFW halo at :math:`l_\max = 8` that would be 6 of the 15 retained modes,
worth 40-47% in those :math:`\Phi_{lm}` near :math:`r_\max` and 1.7% in
:math:`|a|` at :math:`r = 250` with :math:`r_\max = 300`.

Note the inner gate's ``1e-8 * scale`` threshold is deliberately *not*
mirrored here: ``scale`` is the per-mode maximum over the whole radial range,
set by the inner cusp, and is ~9 orders of magnitude larger than
:math:`\rho_{lm}(r_\max)` -- reusing it disables the outer tail entirely.
"""

__all__: tuple[str, ...] = ()


from jaxtyping import Array, Float
from typing import Any

import jax
import numpy as np

import quaxed.numpy as jnp

import galax.potential.custom_types as gt
from .spline import fit_log_spline


def _log_floor(x: Float[Array, "..."], /) -> float:
    r"""Smallest density this module will treat as non-zero.

    Used twice: as a floor inside ``log|rho|`` so an identically-zero mode
    gives an ordinary number rather than ``-inf``, and as a floor on
    ``scale`` so the relative gate cannot divide by zero for an all-zero
    column.

    Sized from the working dtype, not fixed. A float64 constant such as
    ``1e-300`` underflows to *exactly zero* in float32 -- which is what a
    caller gets, since `galax` does not enable x64 on import -- so the floor
    silently stops flooring and an identically-zero mode, routine for
    :math:`l \ge 1`, takes ``log(0)`` and poisons the gradient.

    Sixteen times the smallest normal keeps it clear of denormals while
    staying far below any density a unit system produces.
    """
    return 16.0 * float(np.finfo(x.dtype).tiny)


_SLOPE_TOL: float = 1e-6
"""Floor on the magnitude of a tail's exponent denominator.

The tail integrals divide by an exponent that passes through zero as the
fitted slope crosses the convergence boundary. Clamping keeps the division
finite under ``jit``; the tail is then *dropped* rather than scaled, since a
clamped denominator no longer represents the integral (see
`solve_poisson_profiles`).
"""


def _active_tol(x: Float[Array, "..."], /) -> float:
    r"""Relative threshold for a non-negligible *inner* boundary value.

    ``sqrt(eps)``, which is where the fixed ``1e-8`` came from: that is
    ``sqrt(eps)`` in float64. Taken from the working dtype instead, since in
    float32 ``1e-8`` is 0.08 eps -- below round-off, so the gate degenerates
    from "negligible" to "exactly zero".

    Applies to the inner tail only -- the outer tail gates on
    :math:`\rho_{lm}(r_\max) \neq 0` instead, because this threshold is taken
    against the per-mode maximum over the whole radial range, which is set by
    the inner cusp and would reject every mode at the outer boundary.
    """
    return float(np.sqrt(np.finfo(x.dtype).eps))


def gl_log_nodes(
    log_r: Float[Array, "n_r"], k: int, /
) -> tuple[Float[Array, "n_r-1 k"], Float[Array, "n_r-1 k"]]:
    r"""Gauss-Legendre nodes and weights for :math:`\int \cdot\, d(\log r)`.

    One ``k``-point rule per interval, so the caller can sample a density
    *between* knots. ``k`` is static: the Legendre abscissae come from `numpy`
    at trace time.

    Node positions are affine in ``log_r``, so evaluating this on raw
    :math:`\log r` and on the recentred :math:`\log x` gives the same nodes
    shifted by the same constant -- which is why `solve_poisson_profiles` can
    recompute them internally and still agree with a caller that used this
    helper to place its density samples.
    """
    nodes, weights = np.polynomial.legendre.leggauss(k)
    lo, hi = log_r[:-1, None], log_r[1:, None]
    half = 0.5 * (hi - lo)
    return 0.5 * (lo + hi) + half * nodes, half * weights


@jax.jit
def solve_poisson_profiles(
    r_knots: Float[Array, "n_r"],
    rho_lm: Float[Array, "n_r n_modes"],
    l_per_mode: Float[Array, "n_modes"],
    G: gt.Sz0,
    /,
    rho_gl: Float[Array, "n_r-1 k n_modes"] | None = None,
) -> tuple[
    Float[Array, "n_r n_modes"],
    Float[Array, "n_r n_modes"],
    Float[Array, "n_r n_modes"],
]:
    r"""Solve the radial Poisson equation for every mode.

    ``l_per_mode`` carries :math:`l` as a float per column of ``rho_lm``, in
    the same order. The solve runs as a `jax.lax.scan` over modes rather than
    an unrolled Python loop: one compiled body instead of
    :math:`(l_\max+1)^2` XLA subgraphs, which dominates trace and compile time
    at :math:`l_\max = 8`.

    A `jax.vmap` of the same body is bit-identical and somewhat faster at
    runtime (measured: 81 modes at ``n_r=512``, 2.44 ms -> 1.99 ms; 289
    modes, 7.22 ms -> 4.35 ms) with comparable trace and compile time, so the
    scan is not a runtime optimization over `vmap` -- only over the unrolled
    loop. The gap is a few ms inside a one-off build compile, which is not
    worth the churn of changing it.

    Raises
    ------
    ValueError
        If ``r_knots`` has fewer than 3 entries.
    """
    # Three knots is what the boundary slope fits need. A short grid does not
    # fail on its own: static slicing clamps, so `rho_col[:3]` quietly becomes
    # a 2-point fit and `n_r == 1` returns a meaningless zero. `n_r` is a
    # shape, so this is checked at trace time and costs nothing at runtime.
    if (n_r := r_knots.shape[0]) < 3:
        msg = f"The radial Poisson solve needs at least 3 radial knots, got {n_r}."
        raise ValueError(msg)

    # Shapes are static, so this is a trace-time check that costs nothing at
    # runtime -- and it has to be explicit, because the failure it catches is
    # silent. A `rho_gl` whose interval axis is 1 broadcasts against the
    # `(n_r-1, k)` weights instead of failing, and the solve returns finite,
    # plausible numbers computed from the wrong integrals.
    if rho_gl is not None:
        want = (n_r - 1, rho_gl.shape[1], rho_lm.shape[1])
        if rho_gl.shape != want:
            msg = (
                f"rho_gl must have shape (n_r - 1, k, n_modes) = {want} to "
                f"match r_knots and rho_lm, got {rho_gl.shape}."
            )
            raise ValueError(msg)

    # Work in x = r / r_c, with r_c the log-midpoint of the grid. Every
    # exponent below is then bounded by (l + 2) times *half* the grid's log
    # range instead of (l + 2) |log r|, which is what otherwise overflows:
    # `f_in` carries exp((l+2) log r) while `phi_col` divides it back out by
    # exp(-(l+1) log r), so the intermediate can exceed DBL_MAX even when the
    # ratio is perfectly ordinary, and inf * 0 is a nan. Unreachable in kpc,
    # but |log r| ~ 50 in metres puts l_max ~ 12 within reach of the cliff.
    #
    # The substitution is exact: under r -> e^c r at fixed rho, both
    # r^-(l+1) Int rho r'^(l+2) dr' and r^l Int rho r'^(1-l) dr' pick up
    # exactly e^(2c), so one factor restores the answer at the end.
    log_rc = 0.5 * (jnp.log(r_knots[0]) + jnp.log(r_knots[-1]))
    log_r = jnp.log(r_knots) - log_rc
    # Powers of x are the same for every mode, so they are formed once here
    # rather than inside the scan. Every power the body needs follows from
    # these and `xl` by multiplication, leaving one transcendental per mode
    # instead of four, under the same bound the recentering guarantees.
    x = jnp.exp(log_r)
    x2 = x * x
    du = jnp.diff(log_r)
    du2_12 = du * du / 12.0
    # Gauss-Legendre sampling, when the caller supplied it. The nodes are
    # recomputed here on the recentred `log_r`; `gl_log_nodes` is affine, so
    # these are the caller's nodes shifted by `log_rc`, which is exactly the
    # shift `x` already carries. Weights are interval widths and so are
    # unaffected by the shift.
    if rho_gl is not None:
        log_gl, w_gl = gl_log_nodes(log_r, rho_gl.shape[1])
        e1 = jnp.exp(log_gl)
        e2 = e1 * e1
        e3 = e2 * e1

    # One batched not-a-knot fit for every mode, hoisted out of the scan: the
    # integrands differ per mode only by a power of x, which differentiates
    # exactly, so d(rho x^k)/d(log x) = x^k (drho + k rho). Fitting rho once
    # here instead of each integrand inside the body turns two tridiagonal
    # solves per mode into one solve for all of them -- 32 ms -> 1.8 ms at 81
    # modes, n_r=512 -- and differentiates the smooth profile rather than the
    # steep integrand.
    drho_lm = fit_log_spline(log_r, rho_lm)

    def one_mode(
        _: None,
        xs: tuple[Float[Array, "n_r"], Float[Array, "n_r"], Any, Float[Array, ""]],
    ) -> tuple[None, tuple[Array, Array, Array]]:
        rho_col, drho_col, rho_gl_col, l = xs
        # Named for x, not r: `log_r` is already centred, so this is x^l.
        xl = jnp.exp(l * log_r)  # the only exp per mode
        # Integrands for d(log x), not dx: the dx -> x d(log x) Jacobian is
        # folded in, so these carry one more power of x than the dr form.
        x_in = xl * x2 * x  # x^(l+3)
        x_out = x2 / xl  # x^(2-l)
        g_in = rho_col * x_in
        g_out = rho_col * x_out
        # d(rho x^k)/d(log x), exact in the power, splined in rho.
        d_in = x_in * (drho_col + (l + 3.0) * rho_col)
        d_out = x_out * (drho_col + (2.0 - l) * rho_col)

        def panels(g: Float[Array, "n_r"], d: Float[Array, "n_r"], /) -> Array:
            """Per-interval integral of ``g`` against d(log x).

            The trapezoid plus the endpoint-slope correction that makes it
            exact for cubics.
            """
            out: Array = 0.5 * (g[:-1] + g[1:]) * du - du2_12 * (d[1:] - d[:-1])
            return out

        if rho_gl is not None:
            # Exact for anything the k-point rule integrates, which for a
            # smooth profile is machine precision -- the interpolation error
            # the Hermite rule carries is gone, because rho is *sampled*
            # inside the interval rather than reconstructed across it.
            xl_gl = jnp.exp(l * log_gl)
            p_in = jnp.sum(w_gl * rho_gl_col * xl_gl * e3, axis=1)
            p_out = jnp.sum(w_gl * rho_gl_col * e2 / xl_gl, axis=1)
        else:
            p_in = panels(g_in, d_in)
            p_out = panels(g_out, d_out)

        floor = _log_floor(rho_col)
        scale = jnp.max(jnp.abs(rho_col)) + floor

        # -- inner tail (0 -> r_min), rho_lm ~ A_in r^alpha_in --------------
        log_rho_in = jnp.log(jnp.abs(rho_col[:3]) + floor)
        alpha_in = jnp.mean(jnp.diff(log_rho_in) / jnp.diff(log_r[:3]))
        exp_in = alpha_in + l + 3.0
        safe_in = jnp.where(jnp.abs(exp_in) > _SLOPE_TOL, exp_in, _SLOPE_TOL)
        # The amplitude never appears on its own. Writing the tail as
        # `A_in r_min^exp_in` with `A_in = rho_0 r_min^-alpha_in` would form
        # `exp(-alpha_in log r_min)` first, and `alpha_in` is a three-point
        # slope of a mode that may be pure round-off -- 400+ is routine, so
        # that intermediate overflows to `inf` and `inf * exp(-large)` gives
        # `nan`, poisoning the column and then, through `fit_log_spline`, the
        # whole build. The powers cancel exactly,
        #     A_in r_min^(alpha_in + l + 3) = rho_0 r_min^(l + 3) ,
        # so forming the product directly keeps the exponent bounded by
        # (l + 3) times half the grid's log range, the same bound the
        # recentering above already guarantees. This is also what the outer
        # tail below does.
        dI_in = rho_col[0] * xl[0] * x2[0] * x[0] / safe_in
        # The clamp keeps the division finite under jit, but a clamped
        # denominator no longer represents the integral: at exp_in = 1e-9 the
        # true tail is ~1e3 times what `_SLOPE_TOL` yields. Inside the clamped
        # window the tail is therefore dropped, not scaled -- the same
        # conservative treatment as just across the exp_in <= 0 boundary.
        dI_in = jnp.where(
            (jnp.abs(rho_col[0]) > _active_tol(rho_col) * scale)
            & (exp_in > _SLOPE_TOL),
            dI_in,
            0.0,
        )
        I_in = jnp.concat([jnp.zeros(1), jnp.cumsum(p_in)]) + dI_in

        # -- outer tail (r_max -> inf), rho_lm ~ A_out r^alpha_out ----------
        log_rho_out = jnp.log(jnp.abs(rho_col[-3:]) + floor)
        alpha_out = jnp.mean(jnp.diff(log_rho_out) / jnp.diff(log_r[-3:]))
        active_out = jnp.abs(rho_col[-1]) > 0.0
        denom = l - alpha_out - 2.0
        safe_out = jnp.where(jnp.abs(denom) > _SLOPE_TOL, denom, _SLOPE_TOL)
        dI_out = rho_col[-1] * x2[-1] / xl[-1] / safe_out
        # Same reasoning as the inner tail: `denom` in (0, _SLOPE_TOL] passes
        # the convergence test but is clamped in the division, so drop the
        # tail there rather than under-weight it by an arbitrary factor.
        dI_out = jnp.where(active_out & (denom > _SLOPE_TOL), dI_out, 0.0)
        I_out = jnp.concat([jnp.cumsum(p_out[::-1])[::-1], jnp.zeros(1)]) + dI_out

        # The two integrals carry the derivatives too. Differentiating
        #     Phi = pref (x^-(l+1) I_in + x^l I_out)
        # in log x, the dI/d(log x) terms are +rho x^2 and -rho x^2 and cancel
        # exactly, so dPhi needs no new quadrature; the second derivative
        # keeps one surviving rho x^2, which is just Poisson's equation.
        pref = -4.0 * jnp.pi * G / (2.0 * l + 1.0)
        a_in = I_in / (xl * x)
        a_out = xl * I_out
        phi_col = pref * (a_in + a_out)
        dphi_col = pref * (-(l + 1.0) * a_in + l * a_out)
        d2phi_col = pref * ((l + 1.0) ** 2 * a_in + l * l * a_out) + (
            4.0 * jnp.pi * G * rho_col * x2
        )
        return None, (phi_col, dphi_col, d2phi_col)

    gl_T = (
        jnp.zeros((l_per_mode.size, 0, 0))
        if rho_gl is None
        else jnp.moveaxis(rho_gl, -1, 0)
    )
    _, cols = jax.lax.scan(one_mode, None, (rho_lm.T, drho_lm.T, gl_T, l_per_mode))
    # Every profile scales the same way under the recentring: Phi picks up
    # exp(2 log_rc), and d/d(log x) = d/d(log r) leaves that factor alone.
    scale = jnp.exp(2.0 * log_rc)
    phi, dphi, d2phi = (c.T * scale for c in cols)
    return phi, dphi, d2phi


@jax.jit
def solve_poisson_lm(
    r_knots: Float[Array, "n_r"],
    rho_lm: Float[Array, "n_r n_modes"],
    l_per_mode: Float[Array, "n_modes"],
    G: gt.Sz0,
    /,
) -> Float[Array, "n_r n_modes"]:
    r""":math:`\Phi_{lm}` alone, for callers that do not need the derivatives.

    See `solve_poisson_profiles`, which this delegates to unchanged.
    """
    phi: Float[Array, "n_r n_modes"] = solve_poisson_profiles(
        r_knots, rho_lm, l_per_mode, G
    )[0]
    return phi

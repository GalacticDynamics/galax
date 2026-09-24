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
one requires :math:`|\rho_{lm}(r_\min)|` to exceed ``_ACTIVE_TOL`` times the
per-mode scale, while the outer one requires only
:math:`\rho_{lm}(r_\max) \neq 0`. See "Outer-tail sign" for why the inner
threshold is not reused at the outer boundary.

Choosing ``n_r``
----------------
The interior rule is the trapezoid in :math:`r`, which samples the
:math:`r^{l+2}` factor rather than integrating it, so its error grows with
:math:`l` while :math:`\rho` stays fixed. Measured against the closed form
for :math:`\rho = r^{-1.5}`, interior knots only:

======= ========= ========= =========
l       n_r=256   n_r=512   n_r=1024
======= ========= ========= =========
2       1.8e-3    4.4e-4    1.1e-4
4       5.2e-3    1.3e-3    3.2e-4
8       1.8e-2    4.5e-3    1.1e-3
12      3.8e-2    9.6e-3    2.4e-3
======= ========= ========= =========

So ``n_r`` should rise with ``l_max``: the same accuracy costs roughly twice
the knots for every four levels. A rule that integrates :math:`r^{l+2}`
exactly and interpolates only :math:`\rho` removes the :math:`l` dependence,
but is worse where it matters most -- on a Hernquist monopole it is 4.5x
*less* accurate than the trapezoid, because real :math:`\rho_{lm}` profiles
curve in log-log while the rule assumes they do not.

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

import jax

import quaxed.numpy as jnp

import galax.potential.custom_types as gt

_LOG_FLOOR: float = 1e-300
"""Smallest density this module will treat as non-zero.

Used twice: as a floor inside ``log|rho|`` so an identically-zero mode gives
an ordinary number (~-690) rather than ``-inf``, and as a floor on ``scale``
so the relative gate below cannot divide by zero for an all-zero column.

Well above the smallest normal double (~2.2e-308), and far below any density
a unit system produces, so it never perturbs a real value.
"""

_SLOPE_TOL: float = 1e-6
"""Floor on the magnitude of a tail's exponent denominator.

The tail integrals divide by an exponent that passes through zero as the
fitted slope crosses the convergence boundary. Clamping keeps the division
finite under ``jit``; the tail is then *dropped* rather than scaled, since a
clamped denominator no longer represents the integral (see `solve_poisson_lm`).
"""

_ACTIVE_TOL: float = 1e-8
"""Relative threshold for a non-negligible *inner* boundary value.

Applies to the inner tail only -- the outer tail gates on
:math:`\\rho_{lm}(r_\\max) \\neq 0` instead, because this threshold is taken
against the per-mode maximum over the whole radial range, which is set by the
inner cusp and would reject every mode at the outer boundary.
"""


@jax.jit
def solve_poisson_lm(
    r_knots: Float[Array, "n_r"],
    rho_lm: Float[Array, "n_r n_modes"],
    l_per_mode: Float[Array, "n_modes"],
    G: gt.Sz0,
    /,
) -> Float[Array, "n_r n_modes"]:
    r"""Solve the radial Poisson equation for every mode.

    ``l_per_mode`` carries :math:`l` as a float per column of ``rho_lm``, in
    the same order. The solve runs as a `jax.lax.scan` over modes rather than
    an unrolled Python loop: one compiled body instead of
    :math:`(l_\max+1)^2` XLA subgraphs, which dominates trace and compile time
    at :math:`l_\max = 8`.

    A `jax.vmap` of the same body is bit-identical and in fact somewhat faster
    at runtime (measured: 81 modes at ``n_r=512``, 1.64 ms -> 0.68 ms; 289
    modes, 6.58 ms -> 3.72 ms) with comparable trace and compile time, so the
    scan is not a runtime optimization over `vmap` -- only over the unrolled
    loop. The gap is ~1 ms inside a ~1500 ms one-off build compile, which is
    not worth the churn of changing it.

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
        msg = f"solve_poisson_lm needs at least 3 radial knots, got {n_r}."
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
    dr = jnp.diff(x)

    def one_mode(
        _: None, xs: tuple[Float[Array, "n_r"], Float[Array, ""]]
    ) -> tuple[None, Float[Array, "n_r"]]:
        rho_col, l = xs
        # Named for x, not r: `log_r` is already centred, so this is x^l.
        xl = jnp.exp(l * log_r)  # the only exp per mode
        f_in = rho_col * xl * x2  # rho x^(l+2)
        f_out = rho_col * x / xl  # rho x^(1-l)
        scale = jnp.max(jnp.abs(rho_col)) + _LOG_FLOOR

        # -- inner tail (0 -> r_min), rho_lm ~ A_in r^alpha_in --------------
        log_rho_in = jnp.log(jnp.abs(rho_col[:3]) + _LOG_FLOOR)
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
            (jnp.abs(rho_col[0]) > _ACTIVE_TOL * scale) & (exp_in > _SLOPE_TOL),
            dI_in,
            0.0,
        )
        I_in = (
            jnp.concatenate(
                [jnp.zeros(1), jnp.cumsum(0.5 * (f_in[:-1] + f_in[1:]) * dr)]
            )
            + dI_in
        )

        # -- outer tail (r_max -> inf), rho_lm ~ A_out r^alpha_out ----------
        log_rho_out = jnp.log(jnp.abs(rho_col[-3:]) + _LOG_FLOOR)
        alpha_out = jnp.mean(jnp.diff(log_rho_out) / jnp.diff(log_r[-3:]))
        active_out = jnp.abs(rho_col[-1]) > 0.0
        denom = l - alpha_out - 2.0
        safe_out = jnp.where(jnp.abs(denom) > _SLOPE_TOL, denom, _SLOPE_TOL)
        dI_out = rho_col[-1] * x2[-1] / xl[-1] / safe_out
        # Same reasoning as the inner tail: `denom` in (0, _SLOPE_TOL] passes
        # the convergence test but is clamped in the division, so drop the
        # tail there rather than under-weight it by an arbitrary factor.
        dI_out = jnp.where(active_out & (denom > _SLOPE_TOL), dI_out, 0.0)
        I_out = (
            jnp.concatenate(
                [
                    jnp.cumsum((0.5 * (f_out[:-1] + f_out[1:]) * dr)[::-1])[::-1],
                    jnp.zeros(1),
                ]
            )
            + dI_out
        )

        phi_col = -4.0 * jnp.pi * G / (2.0 * l + 1.0) * (I_in / (xl * x) + xl * I_out)
        return None, phi_col

    _, phi_T = jax.lax.scan(one_mode, None, (rho_lm.T, l_per_mode))
    return phi_T.T * jnp.exp(2.0 * log_rc)  # type: ignore[no-any-return]

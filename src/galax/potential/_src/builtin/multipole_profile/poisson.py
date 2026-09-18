r"""Radial Poisson solve for each spherical harmonic mode.

.. math::

    \Phi_{lm}(r) = -\frac{4\pi G}{2l+1} \left[
        r^{-(l+1)} \int_0^r \rho_{lm}(r') r'^{l+2} dr'
        + r^l \int_r^\infty \rho_{lm}(r') r'^{1-l} dr'
    \right]

Ported from ``bfeax.potential._poisson_scan``.

Boundary tails
--------------
Truncating the grid at ``r_min`` and ``r_max`` biases both integrals. The
power-law slope of :math:`\rho_{lm}` is estimated at each boundary from three
grid points and an analytic tail appended, each guarded for convergence and
for a negligible boundary value.

Outer-tail sign
---------------
The outer-tail gate is sign-agnostic, unlike ``bfeax``, which tests the
signed value and so drops the correction for modes with
:math:`\rho_{lm}(r_\max) < 0` -- routine for :math:`l \ge 1`. For a triaxial
NFW halo at :math:`l_\max = 8` that is 6 of 15 retained modes, worth 40-47% in
those :math:`\Phi_{lm}` near :math:`r_\max` and 1.7% in :math:`|a|` at
:math:`r = 250` with :math:`r_\max = 300`. Reported upstream as
https://github.com/jnibauer/bfeax/issues/1

Note the inner gate's ``1e-8 * scale`` threshold is deliberately *not*
mirrored here: ``scale`` is the per-mode maximum over the whole radial range,
set by the inner cusp, and is ~9 orders of magnitude larger than
:math:`\rho_{lm}(r_\max)` -- reusing it disables the outer tail entirely.
"""

__all__: tuple[str, ...] = ()

import functools as ft

from jaxtyping import Array, Float

import jax

import quaxed.numpy as jnp

import galax.potential.custom_types as gt

_LOG_FLOOR: float = 1e-300
"""Floor inside ``log|rho|``, matching ``bfeax`` exactly (bit-fidelity)."""

_SLOPE_TOL: float = 1e-6
"""Guard against a vanishing exponent denominator, matching ``bfeax``."""

_ACTIVE_TOL: float = 1e-8
"""Relative threshold for a non-negligible boundary value, matching ``bfeax``."""


@ft.partial(jax.jit)
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
    """
    log_r = jnp.log(r_knots)
    dr = jnp.diff(r_knots)

    def one_mode(
        _: None, xs: tuple[Float[Array, "n_r"], Float[Array, ""]]
    ) -> tuple[None, Float[Array, "n_r"]]:
        rho_col, l = xs
        f_in = rho_col * jnp.exp((l + 2.0) * log_r)
        f_out = rho_col * jnp.exp((1.0 - l) * log_r)
        scale = jnp.max(jnp.abs(rho_col)) + _LOG_FLOOR

        # -- inner tail (0 -> r_min), rho_lm ~ A_in r^alpha_in --------------
        log_rho_in = jnp.log(jnp.abs(rho_col[:3]) + _LOG_FLOOR)
        alpha_in = jnp.mean(jnp.diff(log_rho_in) / jnp.diff(log_r[:3]))
        A_in = (
            jnp.sign(rho_col[0]) * jnp.abs(rho_col[0]) * jnp.exp(-alpha_in * log_r[0])
        )
        exp_in = alpha_in + l + 3.0
        safe_in = jnp.where(jnp.abs(exp_in) > _SLOPE_TOL, exp_in, _SLOPE_TOL)
        dI_in = A_in * jnp.exp(exp_in * log_r[0]) / safe_in
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
        dI_out = rho_col[-1] * jnp.exp((2.0 - l) * log_r[-1]) / safe_out
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

        phi_col = (
            -4.0
            * jnp.pi
            * G
            / (2.0 * l + 1.0)
            * (jnp.exp(-(l + 1.0) * log_r) * I_in + jnp.exp(l * log_r) * I_out)
        )
        return None, phi_col

    _, phi_T = jax.lax.scan(one_mode, None, (rho_lm.T, l_per_mode))
    return phi_T.T

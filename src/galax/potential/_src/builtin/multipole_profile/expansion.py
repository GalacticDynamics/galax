r"""Evaluation of a multipole profile expansion.

Function-first, matching `galax.potential._src.builtin.zhao`: the numerics
take a flat ``gt.Params`` dict, and the potential classes build that dict from
their own parameters, so the numerics can be reused without inheriting from a
concrete potential class. Nothing here imports a potential class, and the
mixin that adapts these to one lives with that class, not here.

``p`` carries ``r_knots`` plus the nine arrays `build_expansion` returns:
``phi_lm``, ``dphi_lm``, ``d2phi_lm``, ``phi_asympt_powers``,
``phi_asympt_scales``, ``rho_residual_lm``, ``drho_residual_lm``,
``rho_alpha`` and ``rho_amplitude``, each already stripped to the potential's
unit system. ``d2phi_lm`` is what lets the potential interpolate with the
quintic basis rather than the cubic; the solve returns it for free.
The grid itself is the caller's, so it is not part of what the build returns.
"""

__all__: tuple[str, ...] = ()

import functools as ft

from jaxtyping import Array, Float

import jax
import numpy as np

import quaxed.numpy as jnp

import galax.potential.custom_types as gt
from galax.potential._src.harmonic import (
    eval_log_spline,
    eval_log_spline_asympt,
    real_ylm,
)
from galax.potential._src.utils import safe_vector_norm


def _log_r_and_ylm(
    xyz: gt.BtSz3, l_max: int, keys: tuple[tuple[int, int], ...], /
) -> tuple[Float[Array, "*batch"], Float[Array, "*batch n_modes"]]:
    r"""Split positions into :math:`\log r` and a mode-minor harmonic table."""
    r = safe_vector_norm(xyz)
    uvec = xyz / r[..., None]
    # `real_ylm` is mode-major; the radial splines are mode-minor.
    return jnp.log(r), jnp.moveaxis(real_ylm(l_max, keys, uvec), 0, -1)


@ft.partial(jax.jit, static_argnums=(2, 3))
def expansion_potential(
    p: gt.Params, xyz: gt.BtSz3, l_max: int, keys: tuple[tuple[int, int], ...], /
) -> Float[Array, "*batch"]:
    r""":math:`\Phi = \sum_{lm} \Phi_{lm}(r) Y_{lm}(\hat{q})`.

    Outside ``[r_knots[0], r_knots[-1]]`` each mode continues as the power law
    fitted by `asymptotic_coeffs`, joined to the spline in value and slope.
    Inside the grid the continuation is inactive and this is the interpolant
    alone -- the quintic one, since ``d2phi_lm`` is always passed: the modes
    match in value, slope *and* curvature at every knot.
    """
    log_r, Y = _log_r_and_ylm(xyz, l_max, keys)
    coefs = jnp.concat([p["phi_asympt_powers"], p["phi_asympt_scales"]], axis=1)
    phi_lm = eval_log_spline_asympt(
        jnp.log(p["r_knots"]),
        p["phi_lm"],
        p["dphi_lm"],
        coefs,
        log_r,
        p["d2phi_lm"],
    )
    return jnp.sum(phi_lm * Y, axis=-1)  # type: ignore[no-any-return]


def _ln_huge(x: Float[Array, "..."], /) -> float:
    r"""Largest ``|argument to jnp.exp|`` that cannot overflow in this dtype.

    It bounds what is passed *to* `jnp.exp`, not a power applied to some
    base. Here that argument is ``log|amplitude| + alpha (log r - log r0)``,
    which is why the caller clamps the sum rather than the exponent alone:
    clamping ``alpha (log r - log r0)`` by itself still overflows once the
    amplitude is multiplied back in. `harmonic.asympt`'s copy states the same
    bound as ``|exponent * ln x|``, which is that module's spelling of the
    same quantity.

    ~709.8 in float64 but ~88.7 in float32, and `galax` does not enable x64
    on import, so the bound has to come from the working dtype rather than a
    constant.

    TODO: this is the third dtype-sized limit in the package, after
    `harmonic.poisson`'s floor and gate and `harmonic.asympt`'s own copy of
    this one. They want a single home; importing across modules is worse,
    since each is private to the module that defines it.
    """
    return _LN_HUGE_FRAC * float(np.log(np.finfo(x.dtype).max))


_LN_HUGE_FRAC: float = 0.985
"""Fraction of the overflow exponent `_ln_huge` allows through."""


@ft.partial(jax.jit, static_argnums=(2, 3))
def expansion_density(
    p: gt.Params, xyz: gt.BtSz3, l_max: int, keys: tuple[tuple[int, int], ...], /
) -> Float[Array, "*batch"]:
    r""":math:`\rho = \sum_{lm} \rho_{lm}(r) Y_{lm}(\hat{q})`.

    :math:`\rho_{lm}` is the splined residual plus the analytic inner power
    law, :math:`\mathrm{amplitude} (r/r_0)^\alpha`.
    """
    log_r, Y = _log_r_and_ylm(xyz, l_max, keys)
    log_knots = jnp.log(p["r_knots"])
    log_r0 = log_knots[0]
    # Clamp the query, as the cusp term below is clamped. `eval_log_spline`
    # continues the edge cubic with an unbounded local coordinate, so `s**3`
    # overflows far outside the knots: this returned `nan` from r = 1e20 in
    # float32 and 1e300 in float64. Saturating costs nothing real -- the
    # density outside the knots is already documented as meaningless, and it
    # is not continued the way the potential is -- and it is the difference
    # between one absurd radius and a whole vmapped batch coming back `nan`.
    residual = eval_log_spline(
        log_knots,
        p["rho_residual_lm"],
        p["drho_residual_lm"],
        jnp.clip(log_r, log_knots[0], log_knots[-1]),
    )
    # `exp` overflows at ~88 in float32, which `galax` runs by default, and a
    # steep cusp reaches that at radii a caller can actually pass: an `r^-2`
    # profile gives `inf` at the origin and `nan` far outside the grid. The
    # density is not continued outside the knots anyway -- the value there is
    # already documented as meaningless -- so saturating it loses nothing and
    # keeps a single bad radius from poisoning a whole vmapped batch.
    z = p["rho_alpha"] * (log_r[..., None] - log_r0)
    # Clamp `log|amplitude| + z` -- the log of the whole product -- and
    # rebuild from it. There are two ways to get this wrong and both return
    # `inf` at `r = 0` for a steep cusp: bounding `z` alone lets the
    # amplitude overflow it afterwards, and bounding `z` by
    # `ln_huge - log|amp|` is *looser* than `ln_huge` whenever `|amp| < 1`,
    # so `exp(z)` then overflows on its own.
    amp = p["rho_amplitude"]
    log_amp = jnp.log(jnp.abs(amp) + jnp.finfo(z.dtype).tiny)
    lim = _ln_huge(z)
    background = jnp.sign(amp) * jnp.exp(jnp.clip(log_amp + z, -lim, lim))
    return jnp.sum((residual + background) * Y, axis=-1)  # type: ignore[no-any-return]

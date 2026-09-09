"""Fit SCF expansion coefficients to a particle snapshot."""

__all__ = ["compute_coeffs_discrete"]

import functools as ft

from jaxtyping import Array, Float

import jax
from jax.scipy.special import gammaln

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

from .bfe import phi_nl
from galax.potential._src.builtin.multipole import (
    iter_Ylm,
    scaled_radius_and_direction,
)


@ft.partial(jax.jit, static_argnames=("nmax", "lmax", "compute_var"))
def compute_coeffs_discrete(
    xyz: Float[Array, "N 3"],
    mass: Float[Array, "N"],
    /,
    *,
    nmax: int,
    lmax: int,
    r_s: Float[Array, ""],
    compute_var: bool = False,
) -> tuple[Float[Array, "..."], ...]:
    r"""Compute SCF coefficients from samples of a density distribution.

    Parameters
    ----------
    xyz
        Sample positions, shape ``(N, 3)``. Quantity or bare array.
    mass
        Sample masses, shape ``(N,)``. Quantity or bare array.
    nmax, lmax
        Maximum radial and angular expansion orders. Static.
    r_s
        Scale radius, in the same length unit as ``xyz``.
    compute_var
        Also return the ``(2, 2, nmax+1, lmax+1, lmax+1)`` covariance block.

    Returns
    -------
    Snlm, Tnlm
        Expansion coefficients, shape ``(nmax+1, lmax+1, lmax+1)``, as bare
        arrays in whatever unit ``mass`` was given in. Divide by the scale
        mass before handing them to `SCFPotential`.
    cov
        Only when ``compute_var=True``.

    Notes
    -----
    The per-particle contributions are materialized as a
    ``(nmax+1, lmax+1, lmax+1, N)`` array, so peak memory grows as
    ``nmax * lmax**2 * N``. At ``nmax=12, lmax=6, N=10**6`` that is roughly
    5 GB in float64 -- chunk the particles over several calls and sum the
    results, since the coefficients are a plain sum over ``k``.

    """
    # `xyz` and `r_s` must share a length unit. If `r_s` carries one it sets
    # the scale for both; otherwise every input is taken as a bare array.
    ulen = getattr(r_s, "unit", None)
    xyz = jnp.asarray(u.ustrip(AllowValue, ulen, xyz) if ulen else xyz)
    r_s = jnp.asarray(u.ustrip(AllowValue, ulen, r_s) if ulen else r_s)
    umass = getattr(mass, "unit", None)
    mass = jnp.asarray(u.ustrip(AllowValue, umass, mass) if umass else mass)

    s, uvec = scaled_radius_and_direction(xyz, r_s)

    # shape: nmax+1 by lmax+1 by N
    phinl = phi_nl(nmax, lmax, s)

    # Angular part on the full (l, m) grid: (lmax+1, lmax+1, N)
    shape = (lmax + 1, lmax + 1, len(s))
    cYg, sYg = jnp.zeros(shape), jnp.zeros(shape)
    for l_, m_, cY, sY in iter_Ylm(lmax, uvec):
        cYg = cYg.at[l_, m_].set(cY)
        sYg = sYg.at[l_, m_].set(sY)

    # A_nl, via gammaln so the numerator does not overflow.
    n = jnp.arange(nmax + 1, dtype=float)[:, None]
    l = jnp.arange(lmax + 1, dtype=float)[None, :]
    knl = 0.5 * n * (n + 4 * l + 3) + (l + 1) * (2 * l + 1)
    log_ratio = gammaln(n + 1) + 2 * gammaln(2 * l + 1.5) - gammaln(n + 4 * l + 3)
    anl = (
        -jnp.exp((8 * l + 6) * jnp.log(2.0) + log_ratio)
        / (4 * jnp.pi * knl)
        * (n + 2 * l + 1.5)
    )  # (nmax+1, lmax+1)

    # k_m = 2 - delta_{m0}, shape lmax+1
    km = 2.0 - (jnp.arange(lmax + 1) == 0).astype(float)

    # Per-particle contribution: (nmax+1, lmax+1, lmax+1, N)
    weight = anl[:, :, None, None] * km[None, None, :, None] * mass
    contrib_s = weight * phinl[:, :, None, :] * cYg[None]
    contrib_t = weight * phinl[:, :, None, :] * sYg[None]

    Snlm = jnp.sum(contrib_s, axis=-1)
    Tnlm = jnp.sum(contrib_t, axis=-1)

    if not compute_var:
        return Snlm, Tnlm

    var_s = jnp.sum(contrib_s**2, axis=-1)
    var_t = jnp.sum(contrib_t**2, axis=-1)
    covar = jnp.sum(contrib_s * contrib_t, axis=-1)
    cov = jnp.stack([jnp.stack([var_s, covar]), jnp.stack([covar, var_t])])
    return Snlm, Tnlm, cov

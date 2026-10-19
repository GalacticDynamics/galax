"""Analytic action-angle-frequency transform for the isochrone potential.

Ported term by term from gala's ``isochrone_xv_to_aa`` (Binney & Tremaine 2008,
Sec. 3.5.2, with the erratum to Eq. 3.225), as pure single-point `jax.numpy`.
"""

__all__ = ["_isochrone_aaf"]

from jaxtyping import Array

import jax.numpy as jnp

from ._utils import safe_sqrt

_HALF_PI = jnp.pi / 2


def _F(x: Array, y: Array) -> Array:
    """Gala's ``F``; each branch sees an in-domain argument (NaN-free grads)."""
    hi, lo = y > _HALF_PI, y < -_HALF_PI
    mid = ~(hi | lo)
    y_hi = jnp.where(hi, y, jnp.pi)
    y_lo = jnp.where(lo, y, -jnp.pi)
    y_mid = jnp.where(mid, y, 0.0)
    return jnp.where(
        hi,
        _HALF_PI - jnp.arctan(jnp.tan(_HALF_PI - 0.5 * y_hi) / x),
        jnp.where(
            lo,
            -_HALF_PI + jnp.arctan(jnp.tan(_HALF_PI + 0.5 * y_lo) / x),
            jnp.arctan(x * jnp.tan(0.5 * y_mid)),
        ),
    )


def _isochrone_aaf(
    m_tot: Array, b: Array, G: Array, xyz: Array, v: Array
) -> tuple[Array, Array, Array]:
    """Actions, angles and frequencies of one point in an isochrone potential.

    Unitless. ``b`` is the isochrone scale radius. Angles are in ``[0, 2 pi)``.
    Unbound points (E >= 0) give NaN everywhere except ``J_phi``.
    Degenerate orbits (L = 0, or exactly circular e = 1 at r -> radial) are not
    handled specially.
    """
    GM = G * m_tot
    r2 = xyz @ xyz
    r = safe_sqrt(r2)
    rdotv = xyz @ v
    vr = rdotv / r
    Lvec = jnp.cross(xyz, v)
    Lz = Lvec[2]
    L = safe_sqrt(Lvec @ Lvec)

    E_raw = 0.5 * (v @ v) - GM / (b + jnp.sqrt(b * b + r2))
    bound = E_raw < 0
    E = jnp.where(bound, E_raw, -1.0)  # dummy bound value keeps grads finite

    # --- actions
    sq = jnp.sqrt(L * L + 4 * GM * b)
    Jr = GM / jnp.sqrt(-2 * E) - 0.5 * (L + sq)

    # --- angles
    c = GM / (-2 * E) - b
    e = safe_sqrt(1 - L * L * (1 + b / c) / GM / c)
    eta = jnp.arctan2(r * vr / jnp.sqrt(-2.0 * E), b + c - jnp.sqrt(b * b + r2))
    thetar = eta - e * c * jnp.sin(eta) / (c + b)

    # psi = atan2(cos(theta), -sin(theta) r^2 theta_dot / L), written without
    # sin(theta) in a denominator.
    cos_t = xyz[2] / r
    s = r2 * v[2] - xyz[2] * rdotv
    a1, a2 = cos_t, s / (r * L)
    degenerate = (a1 * a1 + a2 * a2) < 1e-20
    psi = jnp.arctan2(jnp.where(degenerate, 1.0, a1), jnp.where(degenerate, 0.0, a2))

    omega_ratio = 0.5 * (1 + L / sq)
    a = jnp.sqrt((1 + e) / (1 - e))
    ap = jnp.sqrt((1 + e + 2 * b / c) / (1 - e + 2 * b / c))
    A = omega_ratio * thetar - _F(a, eta) - _F(ap, eta) * (L / sq)
    thetat = psi + A

    # sin(u) = Lz cos(theta) / (|L_xy| sin(theta)), clipped to [-1, 1]
    sin_t = safe_sqrt(xyz[0] ** 2 + xyz[1] ** 2) / r
    den = safe_sqrt(Lvec[0] ** 2 + Lvec[1] ** 2) * sin_t
    den_ok = den > 1e-12
    sinu = jnp.where(den_ok, Lz * cos_t / jnp.where(den_ok, den, 1.0), 1.0)
    inside = jnp.abs(sinu) < 1.0
    uu = jnp.where(
        inside, jnp.arcsin(jnp.where(inside, sinu, 0.0)), jnp.sign(sinu) * _HALF_PI
    )
    uu = jnp.where(s < 0, jnp.pi - uu, uu)  # vtheta > 0

    phi = jnp.arctan2(xyz[1], xyz[0])
    thetap = phi - uu + jnp.sign(Lz) * thetat
    angles = jnp.mod(jnp.stack([thetar, thetap, thetat]), 2 * jnp.pi)

    # --- frequencies
    omega_r = (-2 * E) ** 1.5 / GM
    freqs = jnp.stack(
        [omega_r, jnp.sign(Lz) * omega_ratio * omega_r, omega_ratio * omega_r]
    )

    nan = jnp.nan
    actions = jnp.stack(
        [jnp.where(bound, Jr, nan), Lz, jnp.where(bound, L - jnp.abs(Lz), nan)]
    )
    return actions, jnp.where(bound, angles, nan), jnp.where(bound, freqs, nan)

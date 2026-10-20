"""Analytic action-angle-frequency transform for the isochrone potential.

Follows the formulae of gala's ``isochrone_xv_to_aa`` (Binney & Tremaine 2008,
Sec. 3.5.2, with the erratum to Eq. 3.225) as pure single-point `jax.numpy`,
with ``psi`` and ``theta_phi`` rewritten (no ``sin(theta)`` in a denominator;
``theta_phi`` through the longitude of the ascending node) and NaN-safe guards
(double ``where``) at every singular point.
"""

__all__: list[str] = []

import functools as ft

from jaxtyping import Array
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from plum import dispatch

import unxt as u

import galax.coordinates as gc
from ._parse import parse_qpt, parse_w, wrap_result
from ._utils import safe_sqrt
from .result import ActionsAnglesFrequencies
from galax.potential import IsochronePotential

_HALF_PI = jnp.pi / 2
_TWO_PI = 2 * jnp.pi
# Angles need c > _C_TOL b, with c = GM / (-2E) - b (c = 0 only at the origin
# at rest), and 1 - e^2 > _Q_TOL for the orbital plane (L > 0).
_C_TOL = 1e-30
_Q_TOL = 1e-30
# Stand-in for lanes outside the domain of the angle formulas (unbound, or the
# origin at rest): a generic bound point in units of b and sqrt(GM / b).
_X0 = np.array([1.0, 0.5, 0.3])
_V0 = np.array([0.1, 0.4, 0.2])


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


def _delta_energy(GM: Array, b: Array, xyz: Array, v: Array) -> Array:
    """``E - Phi(0) >= 0``, without the cancellation of ``E + GM / 2b``."""
    r2 = xyz @ xyz
    return 0.5 * (v @ v) + GM * r2 / (2 * b * (b + jnp.sqrt(b * b + r2)) ** 2)


def _angles(GM: Array, b: Array, xyz: Array, v: Array) -> Array:
    """Angles of a bound point with ``c > _C_TOL b``."""
    r2 = xyz @ xyz
    s = jnp.sqrt(b * b + r2)
    rdotv = xyz @ v
    Lvec = jnp.cross(xyz, v)
    Lz = Lvec[2]
    L2 = Lvec @ Lvec
    L = safe_sqrt(L2)

    dE = _delta_energy(GM, b, xyz, v)
    D = GM - 2 * b * dE  # = -2 E b > 0
    c = 2 * b * b * dE / D  # = GM / (-2E) - b > 0

    # --- theta_r: finite for every e in [0, 1], incl. radial orbits
    # `where` rather than `clip`: JAX gives `clip` the subgradient 0.5 at a tie,
    # which would halve the gradient at q = 1 (exactly circular orbits).
    q_raw = L2 * (c + b) / (GM * c * c)  # 1 - e^2
    q = jnp.where(q_raw < 0, 0.0, jnp.where(q_raw > 1, 1.0, q_raw))
    circ = q >= 1
    e = jnp.where(circ, 0.0, jnp.sqrt(jnp.where(circ, 0.5, 1 - q)))
    ex = c - r2 / (b + s)  # = b + c - s = c e cos(eta), without cancellation
    ey = rdotv / jnp.sqrt(D / b)  # = c e sin(eta)
    # eta is ill-defined (atan2(0, 0)) for circular orbits; fall back to 0
    eta_ok = (ex * ex + ey * ey) > 1e-24 * c * c
    eta = jnp.arctan2(jnp.where(eta_ok, ey, 0.0), jnp.where(eta_ok, ex, 1.0))
    thetar = eta - e * c * jnp.sin(eta) / (c + b)

    # --- orbital plane: undefined for radial orbits (L = 0)
    radial = q <= _Q_TOL
    Ls = jnp.where(radial, 1.0, L)
    qs = jnp.where(radial, 1.0, q)
    # (a1, a2) = sin(i) (sin psi, cos psi), and sin^2(i) = L_xy^2 / L^2, so a
    # planar orbit (inclination < 1e-10) is where psi and the node are both
    # undefined. There psi = pi/2 (gala's convention) and the node is
    # phi - sgn(Lz) pi/2, so theta_phi = phi + sgn(Lz) A is continuous.
    dL = Lvec[0] ** 2 + Lvec[1] ** 2
    planar = radial | (dL <= 1e-20 * L2)
    # psi = atan2(cos(theta), -sin(theta) r^2 theta_dot / L), written without
    # sin(theta) in a denominator. Continuous in v_theta: gala forces
    # psi = pi/2 for |v_theta| <= 1e-10, so at a z-turning point below the
    # plane this gives -pi/2 where gala gives +pi/2.
    r = jnp.sqrt(jnp.where(r2 > 0, r2, 1.0))
    a1 = xyz[2] / r
    a2 = (r2 * v[2] - xyz[2] * rdotv) / (r * Ls)
    psi = jnp.where(
        planar,
        _HALF_PI,
        jnp.arctan2(jnp.where(planar, 1.0, a1), jnp.where(planar, 0.0, a2)),
    )

    sq = jnp.sqrt(L2 + 4 * GM * b)
    omega_ratio = 0.5 * (1 + L / sq)
    a = (1 + e) / jnp.sqrt(qs)  # sqrt((1 + e) / (1 - e))
    ap = jnp.sqrt((1 + e + 2 * b / c) / (qs / (1 + e) + 2 * b / c))
    A = omega_ratio * thetar - _F(a, eta) - _F(ap, eta) * (L / sq)
    thetaz = psi + A

    # theta_phi = Omega + sgn(Lz) theta_z, Omega the ascending-node longitude
    # (defined on the z axis too). Lz = 0: Omega_phi = 0 and no theta_z term.
    phi_ok = planar & (xyz[0] ** 2 + xyz[1] ** 2 > 0)
    phi = jnp.arctan2(jnp.where(phi_ok, xyz[1], 0.0), jnp.where(phi_ok, xyz[0], 1.0))
    node = jnp.where(
        planar,
        phi - jnp.sign(Lz) * _HALF_PI,
        jnp.arctan2(jnp.where(planar, 0.0, Lvec[0]), jnp.where(planar, 1.0, -Lvec[1])),
    )
    thetap = node + jnp.sign(Lz) * thetaz

    thetaz = jnp.where(radial, 0.0, thetaz)
    thetap = jnp.where(radial, 0.0, thetap)
    angles = jnp.mod(jnp.stack([thetar, thetap, thetaz]), _TWO_PI)
    return jnp.where(angles >= _TWO_PI, angles - _TWO_PI, angles)  # mod rounding


def _isochrone_aaf(
    m_tot: Array, b: Array, G: Array, xyz: Array, v: Array
) -> tuple[Array, Array, Array]:
    """Actions, angles and frequencies of one point in an isochrone potential.

    Unitless. ``b`` is the isochrone scale radius. Angles are in ``[0, 2 pi)``.
    Unbound points (E >= 0) give NaN everywhere except ``J_phi``.

    ``theta_r`` is defined for every bound point except the origin at rest
    (where it is 0); ``theta_z`` and ``theta_phi`` need the orbital plane and
    are 0 for radial orbits (``1 - e^2 <= 1e-30``, i.e. ``L ~ 0``). All
    values and gradients of bound points are finite.
    """
    GM = G * m_tot
    Lvec = jnp.cross(xyz, v)
    Lz = Lvec[2]
    L2 = Lvec @ Lvec
    L = safe_sqrt(L2)

    dE = _delta_energy(GM, b, xyz, v)
    D = GM - 2 * b * dE  # = -2 E b
    bound = D > 0
    # unbound lanes: stand-in E = -GM / 4b keeps the gradients finite
    m2E = jnp.where(bound, D, 0.5 * GM) / b  # -2E

    # --- actions
    sq = jnp.sqrt(L2 + 4 * GM * b)
    Jr = GM / jnp.sqrt(m2E) - 0.5 * (L + sq)

    # --- angles: c = 2 b^2 dE / D > _C_TOL b (excludes the origin at rest)
    regular = bound & (2 * b * dE > _C_TOL * D)
    # stand-in point cast to the input dtype, so float32 inputs stay float32
    xyz_s = jnp.where(regular, xyz, b * jnp.asarray(_X0, dtype=xyz.dtype))
    v_s = jnp.where(regular, v, jnp.sqrt(GM / b) * jnp.asarray(_V0, dtype=v.dtype))
    angles = jnp.where(regular, _angles(GM, b, xyz_s, v_s), 0.0)

    # --- frequencies
    omega_ratio = 0.5 * (1 + L / sq)
    omega_r = m2E**1.5 / GM
    freqs = jnp.stack(
        [omega_r, jnp.sign(Lz) * omega_ratio * omega_r, omega_ratio * omega_r]
    )

    nan = jnp.nan
    actions = jnp.stack(
        [jnp.where(bound, Jr, nan), Lz, jnp.where(bound, L - jnp.abs(Lz), nan)]
    )
    return actions, jnp.where(bound, angles, nan), jnp.where(bound, freqs, nan)


@ft.partial(jax.jit, inline=True)
def _aaf(
    pot: IsochronePotential, q: Array, p: Array, t: Array
) -> tuple[Array, Array, Array]:
    """Unitless batched transform; ``q``, ``p`` (..., 3) and ``t`` (...)."""
    usys = pot.units
    batch = q.shape[:-1]
    tq = u.Q(t, usys["time"])
    m_tot = jnp.broadcast_to(pot.m_tot(tq, ustrip=usys["mass"]), batch)
    b = jnp.broadcast_to(pot.r_s(tq, ustrip=usys["length"]), batch)
    G = pot.constants["G"].ustrip(
        usys["length"] ** 3 / (usys["mass"] * usys["time"] ** 2)
    )

    def f(m: Array, b_: Array, x: Array, v: Array) -> tuple[Array, Array, Array]:
        return _isochrone_aaf(m, b_, G, x, v)

    for _ in batch:
        f = jax.vmap(f)
    return f(m_tot, b, q, p)


@dispatch
def isochrone_actions_angles_freqs(
    pot: IsochronePotential, q: Any, p: Any, t: Any, /
) -> ActionsAnglesFrequencies:
    """Actions, angles and frequencies from positions, velocities and time."""
    q, p, t, as_quantity = parse_qpt(pot, q, p, t)
    return wrap_result(*_aaf(pot, q, p, t), pot, as_quantity)


@dispatch
def isochrone_actions_angles_freqs(
    pot: IsochronePotential, w: gc.AbstractPhaseSpaceCoordinate, /
) -> ActionsAnglesFrequencies:
    """Actions, angles and frequencies of a phase-space coordinate or orbit."""
    q, p, t = parse_w(pot, w)
    return wrap_result(*_aaf(pot, q, p, t), pot, as_quantity=True)

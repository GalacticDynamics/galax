"""Staeckel-fudge prolate core (per point, then ``vmap``).

Prolate coordinates written as signed roots, the integrals of motion, and
the F(tau) closures. The turning points are in `_staeckel_limits`.
"""

__all__: list[str] = []

from collections.abc import Callable
from jaxtyping import Array
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._utils import safe_div, safe_sqrt

# ---------------------------------------------------------------------------
# Prolate coordinates, integrals of motion, F(tau).
#
# Everything here is per point; `phi(R, z)` is the potential at (R, 0, z),
# i.e. ``pot._potential(jnp.stack([R, 0*R, z], axis=-1), t)``.

LOG_TAU_MAX = 230.0  # overflow guard on the lambda_max lane (C++: tau < 1e100)


def _sqrt0(x: Array) -> Array:
    """sqrt(max(x, 0)) with a zero (not NaN) derivative where x <= 0."""
    pos = x > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.0)), 0.0)


OFF_AXIS = 1e-30  # Phi is never evaluated at R < OFF_AXIS Delta


def _off_axis(
    phi: Callable[[Array, Array], Array], D2: Array
) -> Callable[[Array, Array], Array]:
    """``phi`` with R moved to OFF_AXIS Delta where it is smaller.

    On the z-axis at z = 0 the core would evaluate Phi at the origin (the
    phase point, Phi(rho, 0) with rho = 0 inside the foci, the lambda-side
    limit of F'(D2) at z = 0, F's branches at tau = D2 or u = 0 there), where
    a cuspy Phi (NFW, MW2022's halo) has NaN derivatives; in reverse mode
    even a masked lane's NaN poisons the rest. 1e-30 Delta changes Phi by
    O(1e-30 Delta / scale), far below roundoff, and R's own tangent on the
    axis is 0 (`_sqrt0`), so values and gradients are unchanged.
    """
    r0 = OFF_AXIS * jnp.sqrt(D2)
    return lambda R, z: phi(jnp.where(r0 < R, R, r0), z)


class Prolate(NamedTuple):
    """Prolate spheroidal coordinates of (R, z) for focal distance^2 D2."""

    lam: Array  # lambda >= D2
    absnu: Array  # |nu| in [0, D2]
    sigma: Array  # sign(z) sqrt|nu|
    dmn: Array  # D2 - |nu|
    lmd: Array  # lambda - D2
    rho: Array  # sqrt(lmd), gradient-safe at R = 0


def prolsph(R: Array, z: Array, D2: Array) -> Prolate:
    """Gala's ``cyl_to_prolsph``, written branchless and gradient-safe.

    Each unselected branch sees masked inputs, so z = 0 and R = 0 (off the
    focal points) give NaN-free derivatives. gala's exactness overrides of
    sqD are not ported (algebraically identical; a gradient hazard).
    """
    R2, z2 = R * R, z * z
    sm, dif = R2 + z2 + D2, R2 + z2 - D2
    sqD = _sqrt0(dif**2 + 4 * R2 * D2)  # 0 at the focal points (R = 0, |z| = Delta)
    pos = dif >= 0
    # dif >= 0: lmd direct, dmn = D2 R2 / lmd; dif < 0: the reverse.
    lmd_p = jnp.where(pos, 0.5 * (sqD + dif), 1.0)
    dmn_n = jnp.where(pos, 1.0, 0.5 * (sqD - dif))
    lmd = jnp.where(pos, lmd_p, D2 * R2 / dmn_n)
    dmn = jnp.where(pos, safe_div(D2 * R2, lmd_p), dmn_n)
    rho = jnp.where(pos, safe_sqrt(lmd_p), R * jnp.sqrt(D2 / dmn_n))

    absnu = 2 * D2 / (sm + sqD) * z2
    big = 2 * absnu > D2
    absnu = jnp.where(big, D2 - dmn, absnu)
    dmn = jnp.where(big, dmn, D2 - absnu)
    signz = jnp.where(z >= 0, 1.0, -1.0)
    sigma = jnp.where(
        big,
        signz * jnp.sqrt(jnp.where(big, D2 - dmn, D2)),
        z * jnp.sqrt(2 * D2 / (sm + sqD)),
    )
    return Prolate(lam=D2 + lmd, absnu=absnu, sigma=sigma, dmn=dmn, lmd=lmd, rho=rho)


class IOM(NamedTuple):
    """Integrals of motion, phase-point coordinates, and their rates."""

    E: Array
    Lz: Array
    I3: Array
    flam: Array
    fnu: Array
    lamdot: Array
    sigdot: Array
    lam: Array
    absnu: Array
    sigma: Array
    dmn: Array
    lmd: Array
    rho: Array
    D2: Array  # Delta^2; 1 where invalid
    valid: Array  # finite phase point and D2 > 0 (else evaluated at SAFE_W6)


# Stand-in phase point and D2 of an invalid lane: off the axis, the plane and
# the focal points, so every derivative there is finite.
SAFE_W6 = (1.0, 0.0, 0.5, 0.0, 0.0, 0.0)
# Scale guards on squared lengths: D2 below TINY_D2 max(R^2 + z^2, 1) is
# lost in roundoff against the coordinates, or its powers in the
# derivatives overflow (from ~1e-200 relative, ~1e-150 absolute), and
# R^2 + z^2 + D2 above MAX_R2 overflows F's squared terms (~1e154).
# The max(..., 1) makes TINY_D2 an absolute floor (in the potential's length
# units^2) for points within unit distance: it is unit-dependent there.
TINY_D2 = 1e-100
MAX_R2 = 1e100


def integrals_of_motion(
    phi: Callable[[Array, Array], Array], w6: Array, D2: Array
) -> IOM:
    """E, L_z, I_3 and f_lambda, f_nu as in the C++, at one point ``w6``.

    lambda-dot and sigma-dot come from one ``jax.jvp`` of `prolsph`. R = 0
    takes the C++ convention v_R = |v_xy| (NaN-free); I_3 is written in
    Cartesian velocities so its (x, y) gradient is right on the axis.

    A non-finite ``w6`` or ``D2``, ``D2 <= 0`` (the pipeline Delta at the
    origin), a D2 below TINY_D2 max(R^2 + z^2, 1), or R^2 + z^2
    + D2 above MAX_R2, is evaluated at `SAFE_W6` with D2 = 1 and flagged
    ``valid = False``: `find_limits` returns NaN there (C++: ``!(delta >
    0)``) and the gradients are finite (zero) rather than NaN.

    The core needs float64 (``jax_enable_x64``): its offsets D2 (1 -+ 1e-15),
    MINIMUM_RANGE = 1e-10 and the exp(230) guard are below float32's
    resolution or range. A floating ``w6`` or ``D2`` of any other precision
    raises a TypeError (it is not upcast: Phi would still be evaluated in it).
    """
    w6, D2 = jnp.asarray(w6), jnp.asarray(D2)
    for x in (w6, D2):
        if jnp.issubdtype(x.dtype, jnp.floating) and x.dtype != jnp.float64:
            msg = f"the Staeckel core needs float64 (jax_enable_x64), got {x.dtype}"
            raise TypeError(msg)
    r2 = w6[0] ** 2 + w6[1] ** 2 + w6[2] ** 2
    valid = (
        jnp.all(jnp.isfinite(w6))
        & (TINY_D2 * jnp.where(r2 > 1, r2, 1.0) < D2)  # also False for NaN
        & (r2 + D2 < MAX_R2)  # also False for inf
    )
    w6 = jnp.where(valid, w6, jnp.asarray(SAFE_W6, dtype=w6.dtype))
    D2 = jnp.where(valid, D2, 1.0)
    phi = _off_axis(phi, D2)
    x, y, z, vx, vy, vz = w6
    R2 = x * x + y * y
    axis = R2 == 0
    R = _sqrt0(R2)
    Rs = jnp.where(axis, 1.0, R)
    Lz = x * vy - y * vx
    vR = jnp.where(axis, _sqrt0(vx * vx + vy * vy), (x * vx + y * vy) / Rs)

    p, dp = jax.jvp(lambda R_, z_: prolsph(R_, z_, D2), (R, z), (vR, vz))
    Phi = phi(R, z)
    Phi0 = phi(p.rho, jnp.zeros_like(z))
    E = Phi + 0.5 * (vx * vx + vy * vy + vz * vz)
    flam = -p.lam * Phi0
    fnu = flam + Phi * (p.lam - p.absnu)
    # C++: (z v_phi)^2 + (R v_z - z v_R)^2. With v_R^2 + v_phi^2 = |v_xy|^2 and
    # R v_R = x v_x + y v_y this is smooth in (x, y), also through R = 0.
    I3 = p.lam * (Phi - Phi0) + 0.5 * (
        z * z * (vx * vx + vy * vy)
        - 2 * z * vz * (x * vx + y * vy)
        + R2 * vz * vz
        + vz * vz * D2
    )
    return IOM(
        E,
        Lz,
        I3,
        flam,
        fnu,
        dp.lam,
        dp.sigma,
        p.lam,
        p.absnu,
        p.sigma,
        p.dmn,
        p.lmd,
        p.rho,
        D2,
        valid,
    )


def _tau_ok(tau: Array) -> Array:
    """Return the one tau-validity predicate: finite and >= 0.

    [0, inf) is the union of both branches' domains, so a tau that rounds
    across D2 still evaluates; anything else (NaN, +-inf, < 0: frozen or
    invalid lanes) is replaced by a safe point before evaluation and the
    result is NaN, which keeps gradients on the other lanes NaN-free.
    """
    return jnp.isfinite(tau) & (tau >= 0)


def _F_val(iom: IOM, tau: Array, Phi: Array, mult: Array, I: Array) -> Array:
    """C++ ``F_tau`` value from the branch coordinates; NaN for bad tau."""
    ok = _tau_ok(tau)
    t = jnp.where(ok, tau, iom.D2)
    F = (iom.E * t - I - Phi * mult) * (t - iom.D2) - iom.Lz**2 / 2 * t
    return jnp.where(ok, F, jnp.nan)


def _semi_tau(s: Array, lam_pos: Array) -> tuple[Array, Array, Array]:
    """Map s to tau = lam_pos exp(s/(1-s)) behind the log tau > 230 guard.

    Returns (tau, beyond, ok): ``beyond`` marks s past the guard (or s >= 1),
    where tau is replaced by lam_pos so exp never overflows; ``ok`` is
    ``isfinite(s)``.
    """
    s = jnp.asarray(s)
    ok = jnp.isfinite(s)
    inner = ok & (s < 1)
    ss = jnp.where(inner, s, 0.0)
    logt = jnp.log(lam_pos) + ss / (1 - ss)
    beyond = (logt > LOG_TAU_MAX) | (ok & ~inner)
    return jnp.exp(jnp.where(beyond, jnp.log(lam_pos), logt)), beyond, ok


def _F_semi_val(iom: IOM, tau: Array, Phi: Array, mult: Array, I: Array) -> Array:
    """F/tau^2 without forming F or tau^2 (cannot overflow); NaN for bad tau."""
    ok = jnp.isfinite(tau) & (tau > 0)
    t = jnp.where(ok, tau, iom.D2)
    F = (iom.E - I / t - Phi * mult / t) * (1 - iom.D2 / t) - iom.Lz**2 / (2 * t)
    return jnp.where(ok, F, jnp.nan)


class Fset(NamedTuple):
    """F(tau) closures for one point (see `make_F`)."""

    F_lam: Callable[[Array], Array]
    F_nu_u: Callable[[Array], Array]
    F_nu: Callable[[Array], Array]
    lane_coords: Callable[[Array], tuple[Array, Array, Array, Array]]
    dF_lam: Callable[[Array], Array]  # noqa: N815
    d2F_lam: Callable[[Array], Array]  # noqa: N815
    dF_nu: Callable[[Array], Array]  # noqa: N815
    F_semi: Callable[[Array, Array, Array], Array]
    F_noise: Callable[[Array], tuple[Array, Array]]  # (F, its roundoff scale)
    phi: Callable[[Array, Array], Array]  # the potential the closures use
    D2: Array  # focal distance^2
    rho: Array  # R of the nu-line at u = 0 (z = 0): sqrt(lam - D2)


def make_F(phi: Callable[[Array, Array], Array], iom: IOM) -> Fset:  # noqa: C901
    """F(tau) on the lambda and nu branches, its tau-derivatives and F/tau^2.

    - lambda branch: z = sqrt(tau) sigma / Delta, R = sqrt(tau-D2) sqrt(dmn) / Delta.
    - nu branch, through the signed root u of tau: z = sqrt(lam) u / Delta,
      R = rho sqrt(D2-u^2) / Delta.

    tau outside `_tau_ok` (non-finite or negative) gives NaN, with NaN-free
    gradients on the other lanes. ``F_semi(s, lam_pos, phi_inf)`` is F/tau^2 at
    tau = lam_pos exp(s/(1-s)), and E - phi_inf beyond log tau > 230.
    """
    D2 = iom.D2
    phi = _off_axis(phi, D2)
    Delta = jnp.sqrt(D2)
    sdmn = safe_sqrt(iom.dmn)
    sqlam = jnp.sqrt(iom.lam)
    I_lam, I_nu = iom.I3 - iom.fnu, iom.I3 - iom.flam
    one = jnp.ones_like
    # Phi on the axis at tau = D2 of each branch, for F'(D2)'s limits: once
    Phi_lam_D2 = phi(jnp.zeros_like(iom.sigma), iom.sigma)
    Phi_nu_D2 = phi(jnp.zeros_like(sqlam), sqlam)

    def lam_coords(t: Array) -> tuple[Array, Array, Array, Array]:
        R = _sqrt0(t - D2) * sdmn / Delta
        z = safe_sqrt(t) * iom.sigma / Delta  # t = 0 is valid (off-branch)
        return R, z, t - iom.absnu, jnp.broadcast_to(I_lam, t.shape)

    def nu_coords(u: Array) -> tuple[Array, Array, Array, Array]:
        R = iom.rho * _sqrt0(D2 - u * u) / Delta
        z = sqlam * u / Delta
        return R, z, u * u - iom.lam, jnp.broadcast_to(I_nu, u.shape)

    def lane_coords(tau: Array) -> tuple[Array, Array, Array, Array]:
        tau = jnp.asarray(tau)
        ok = _tau_ok(tau)
        is_lam = tau >= D2
        Rl, zl, ml, Il = lam_coords(jnp.where(ok & is_lam, tau, 2 * D2))
        Rn, zn, mn, In = nu_coords(safe_sqrt(jnp.where(ok & ~is_lam, tau, 0.5 * D2)))
        return (
            jnp.where(is_lam, Rl, Rn),
            jnp.where(is_lam, zl, zn),
            jnp.where(is_lam, ml, mn),
            jnp.where(is_lam, Il, In),
        )

    def F_lam(tau: Array) -> Array:
        tau = jnp.asarray(tau)
        ok = _tau_ok(tau)
        t = jnp.where(ok, tau, 2 * D2)
        R, z, mult, I = lam_coords(t)
        return jnp.where(ok, _F_val(iom, t, phi(R, z), mult, I), jnp.nan)

    def F_nu_u(u: Array) -> Array:
        u = jnp.asarray(u, dtype=jnp.result_type(u, iom.D2, float))
        ok = jnp.isfinite(u)
        us = jnp.where(ok, u, 0.0)
        R, z, mult, I = nu_coords(us)
        return jnp.where(ok, _F_val(iom, us * us, phi(R, z), mult, I), jnp.nan)

    def F_nu(tau: Array) -> Array:
        tau = jnp.asarray(tau)
        ok = _tau_ok(tau)
        return jnp.where(ok, F_nu_u(safe_sqrt(jnp.where(ok, tau, 0.0))), jnp.nan)

    def dF_lam(tau: Array) -> Array:
        tau = jnp.asarray(tau)
        ok = _tau_ok(tau)
        at = tau == D2  # sqrt(tau - D2): use the lambda-side limit
        t = jnp.where(ok & ~at, tau, 2 * D2)
        d = jax.jvp(F_lam, (t,), (one(t),))[1]
        lim = iom.E * D2 - iom.Lz**2 / 2 - I_lam - (D2 - iom.absnu) * Phi_lam_D2
        return jnp.where(ok, jnp.where(at, lim, d), jnp.nan)

    def d2F_lam(tau: Array) -> Array:
        tau = jnp.asarray(tau)
        ok = jnp.isfinite(tau) & (tau > D2)  # NaN at D2, as the C++'s 0/0
        t = jnp.where(ok, tau, 2 * D2)
        d2 = jax.jvp(lambda x: jax.jvp(F_lam, (x,), (one(x),))[1], (t,), (one(t),))[1]
        return jnp.where(ok, d2, jnp.nan)

    def dF_nu(tau: Array) -> Array:
        tau = jnp.asarray(tau)
        ok = _tau_ok(tau)
        at0, atD = tau == 0, tau == D2
        gen = ok & ~at0 & ~atD
        u = jnp.sqrt(jnp.where(gen, tau, 0.25 * D2))
        u_ev = jnp.where(at0, 0.0, u)
        Fu, Fuu = jax.jvp(
            lambda x: jax.jvp(F_nu_u, (x,), (one(x),))[1], (u_ev,), (one(u_ev),)
        )
        d = jnp.where(at0, Fuu / 2, Fu / (2 * u))  # u = 0: F'(tau) = F_uu(0)/2
        lim = iom.E * D2 - iom.Lz**2 / 2 - I_nu - (D2 - iom.lam) * Phi_nu_D2
        return jnp.where(ok, jnp.where(atD, lim, d), jnp.nan)

    def F_noise(tau: Array) -> tuple[Array, Array]:
        # F on either branch and eps (S |tau - D2| + L_z^2 tau / 2), S =
        # |E tau| + |I| + |Phi mult| the size of F's terms: one phi call.
        tau = jnp.asarray(tau)
        R, z, mult, I = lane_coords(tau)
        Phi = phi(R, z)
        S = jnp.abs(iom.E * tau) + jnp.abs(I) + jnp.abs(Phi * mult)
        eps = jnp.finfo(jnp.result_type(tau, D2)).eps  # type: ignore[no-untyped-call]
        noise = eps * (S * jnp.abs(tau - D2) + iom.Lz**2 * jnp.abs(tau) / 2)
        return _F_val(iom, tau, Phi, mult, I), noise

    def F_semi(s: Array, lam_pos: Array, phi_inf: Array) -> Array:
        tau, beyond, ok = _semi_tau(s, lam_pos)
        R, z, mult, I = lane_coords(tau)
        val = _F_semi_val(iom, tau, phi(R, z), mult, I)
        return jnp.where(ok, jnp.where(beyond, iom.E - phi_inf, val), jnp.nan)

    return Fset(
        F_lam,
        F_nu_u,
        F_nu,
        lane_coords,
        dF_lam,
        d2F_lam,
        dF_nu,
        F_semi,
        F_noise,
        phi,
        D2,
        iom.rho,
    )

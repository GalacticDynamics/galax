"""Test helpers for the action-angle tests.

A separable and a scale-safe Kuzmin-Kutuzov potential, the KK focal
distance, a numpy bisection, and the shared fixtures of the Staeckel core
tests.
"""

__all__ = [
    "ScaledKuzminKutuzovPotential",
    "SeparablePotential",
    "bisect",
    "check_circular_fwd_rev",
    "check_dF_limit_sites",
    "check_grad_safety",
    "kk_delta",
    "kk_turning_points",
]

import functools as ft

import jax
import jax.numpy as jnp
import numpy as np

import quaxed.numpy as qnp
import unxt as u
from unxt.quantity import AllowValue

import galax.potential as gp
from galax.actionangle._src import (
    _staeckel_limits as limits_mod,
    _staeckel_prolate,
    _utils,
)
from galax.actionangle._src._staeckel_limits import (
    attach_tangents,
    default_phi_inf,
    find_limits,
    taylor_coeffs,
)
from galax.actionangle._src._staeckel_prolate import (
    integrals_of_motion,
    make_F,
)
from galax.actionangle._src.staeckel import _delta
from galax.potential.params import AbstractParameter, ParameterField


class ScaledKuzminKutuzovPotential(gp.AbstractSinglePotential):
    r"""Kuzmin-Kutuzov, Delta^2 = a^2 - c^2, evaluated with lengths in units of c.

    `galax.potential.KuzminKutuzovPotential` squares the input coordinates,
    which overflows float32 in SI units (R ~ 1e20 m); this one does not, to
    test that `staeckel_fudge_delta` itself is scale-free there.
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    @jax.jit
    def _potential(self, xyz, t, /):
        t = u.Q.from_(t, self.units["time"])
        G = self.constants["G"].value
        m = self.m_tot(t, ustrip=self.units["mass"])
        a = self.a(t, ustrip=self.units["length"])
        c = self.c(t, ustrip=self.units["length"])
        xyz = u.ustrip(AllowValue, self.units["length"], xyz) / c
        R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
        z2 = xyz[..., 2] ** 2
        a2 = (a / c) ** 2
        lam_nu = R2 + a2 * z2 + a2
        denom = jnp.sqrt(R2 + z2 + a2 + 1 + 2 * jnp.sqrt(lam_nu))
        return -G * m / c / denom


class SeparablePotential(gp.AbstractSinglePotential):
    r"""$\Phi = f(R) + g(z)$, so $\partial^2\Phi / \partial R \partial z = 0$.

    $f = \frac{1}{2} R^2$, $g = z^2$, in the potential's units.
    """

    @jax.jit
    def _potential(self, xyz, t, /):
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        return 0.5 * (xyz[..., 0] ** 2 + xyz[..., 1] ** 2) + xyz[..., 2] ** 2


def bisect(f, a, b, n_iter=200):
    """Root of ``f`` in [a, b] (f(a) f(b) <= 0) by bisection, to the last ulp.

    Vectorized: ``a`` and ``b`` may be arrays, with ``f`` evaluated on all
    lanes at once. Stops changing a lane once its midpoint rounds to an end.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    fa = np.asarray(f(a))
    for _ in range(n_iter):
        m = 0.5 * (a + b)
        if np.all((m == a) | (m == b)):
            break
        fm = np.asarray(f(m))
        left = np.sign(fm) == np.sign(fa)  # root in [m, b]
        a, fa = np.where(left, m, a), np.where(left, fm, fa)
        b = np.where(left, b, m)
    return 0.5 * (a + b)


def kk_delta(pot: gp.KuzminKutuzovPotential, t: float = 0.0) -> float:
    """Focal distance Delta = sqrt(a^2 - c^2) at time ``t``, in ``pot.units``."""
    t = u.Q(t, pot.units["time"])
    a = pot.a(t, ustrip=pot.units["length"])
    c = pot.c(t, ustrip=pot.units["length"])
    return float(qnp.sqrt(a**2 - c**2))


def kk_turning_points(
    pot: gp.KuzminKutuzovPotential, w6: np.ndarray
) -> tuple[float, float, float]:
    """Exact (lam_min, lam_max, nu_max) of a KK orbit at t=0, in numpy.

    KK is Staeckel with G(tau) = GM sqrt(tau + c^2): Phi = -(G(lam) - G(nu)) /
    (lam - nu), and both momenta vanish where F(tau) = (E tau + G(tau) - K)
    (tau - D2) - Lz^2 tau / 2 = 0. The separation constant K follows from
    F(lam) = (lam - nu)^2 lamdot^2 / (8 lam). Needs Lz != 0 and R > 0.
    """
    t = u.Q(0.0, pot.units["time"])
    GM = float(pot.constants["G"].value * pot.m_tot(t, ustrip=pot.units["mass"]))
    c2 = float(pot.c(t, ustrip=pot.units["length"])) ** 2
    D2 = kk_delta(pot) ** 2
    x, y, z, vx, vy, vz = (float(v) for v in np.asarray(w6))
    R = np.hypot(x, y)
    vR, Lz = (x * vx + y * vy) / R, x * vy - y * vx
    S, dif = R * R + z * z + D2, R * R + z * z - D2
    sqD = np.sqrt(dif**2 + 4 * R * R * D2)
    # lam - D2 without cancellation (it rounds to 0 for R ~ 1e-8 |z| < Delta)
    lmd = 0.5 * (sqD + dif) if dif >= 0 else 2 * D2 * R * R / (sqD - dif)
    lam, nu = D2 + lmd, 2 * D2 * z * z / (S + sqD)
    lamdot = (2 * R * lam * vR + 2 * z * lmd * vz) / sqD

    def G(tau: float) -> float:
        return GM * np.sqrt(tau + c2)

    E = -(G(lam) - G(nu)) / (lam - nu) + 0.5 * (vx * vx + vy * vy + vz * vz)
    Flam = (lam - nu) ** 2 * lamdot**2 / (8 * lam)
    K = E * lam + G(lam) - (Flam + Lz**2 * lam / 2) / lmd

    def F(tau: float) -> float:
        return (E * tau + G(tau) - K) * (tau - D2) - Lz**2 * tau / 2

    hi = 2 * lam
    while F(hi) >= 0:
        hi *= 2
    # F(lam) ~ 0 by roundoff near a turning point (or lam = D2): split the
    # brackets at the first point above lam where F > 0
    grid = lam + (hi - lam) * np.logspace(-14, 0, 300)
    mid = grid[np.argmax([F(t) > 0 for t in grid])]
    return (
        float(bisect(F, D2, mid)),
        float(bisect(F, mid, hi)),
        float(bisect(F, nu, D2)),
    )


# ---------------------------------------------------------------------------
# Shared fixtures of the Staeckel core tests (prolate, limits, tangents, Taylor).

T0 = jnp.asarray(0.0)


def _kk():
    return gp.KuzminKutuzovPotential(
        m_tot=u.Q(1e11, "Msun"), a=u.Q(3.0, "kpc"), c=u.Q(1.0, "kpc"), units="galactic"
    )


def _mn():
    return gp.MiyamotoNagaiPotential(
        m_tot=u.Q(1e11, "Msun"), a=u.Q(3.0, "kpc"), b=u.Q(0.3, "kpc"), units="galactic"
    )


def _nfw():
    return gp.NFWPotential(m=u.Q(1e12, "Msun"), r_s=u.Q(15.0, "kpc"), units="galactic")


def _mn_p(p):
    return gp.MiyamotoNagaiPotential(
        m_tot=u.Q(p[0], "Msun"),
        a=u.Q(p[1], "kpc"),
        b=u.Q(p[2], "kpc"),
        units="galactic",
    )


# Gradient-safety potentials. NFW and MW2022 (NFW halo) are cuspy: their own
# derivatives are NaN at the origin, so the core must never evaluate them there.
def _composite():
    """Make a small Milky-Way-like composite: a Miyamoto-Nagai disk, NFW halo.

    MW2022 has six components; its lambda Taylor nest differentiated in
    jacfwd or jacrev takes ~2.5 GB to compile.
    """
    return gp.CompositePotential(disk=_mn(), halo=_nfw())


_GRAD_POTS = {
    "kk": _kk,
    "mn": _mn,
    "nfw": _nfw,
    "mw2022": gp.MilkyWayPotential2022,
    "composite": _composite,
}


def _phi(pot, counter=None):
    def phi(R, z):
        if counter is not None:
            counter.append(1)
        xyz = jnp.stack([R, jnp.zeros_like(R), z], axis=-1)
        return pot._potential(xyz, T0)

    return phi


D2 = kk_delta(_kk()) ** 2  # 8


MN_P = jnp.asarray([1e11, 3.0, 0.3])


MN_D2 = 4.0  # an explicit, fixed Delta^2: gradients are at fixed Delta


# On the z-axis (R = 0) inside the foci (|z| < Delta); L_z = 0.
AXIS_W6 = jnp.asarray([0.0, 0.0, 1.0, 0.05, 0.01, 0.02])


# z = 0 with v_z != 0 (sigma = 0): the lambda-side F'(D2) limit term sits at
# Phi(0, sigma), the origin, unless masked.
MID_W6 = jnp.asarray([5.0, 0.0, 0.0, 0.05, 0.2, 0.03])


def _w6s(n=40, seed=1):
    """Bound, generic phase-space points (off the KK focal points)."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.5, 8.0, size=(n, 3))
    q[:, 2] *= rng.choice([-1, 1], size=n) * 0.6
    v = rng.normal(0.0, 0.08, size=(n, 3))
    v[:, 1] += 0.15  # rotation
    return jnp.asarray(np.concat([q, v], axis=1))


def _setup(pot=None, w6=None, counter=None):
    pot = _kk() if pot is None else pot
    phi = _phi(pot, counter)
    w6 = _w6s(1)[0] if w6 is None else w6
    iom = integrals_of_motion(phi, w6, D2)
    return phi, iom, make_F(phi, iom)


# Jitted programs, shared across tests (a compile costs far more than a run):
# keyed on whatever the program closes over (see `_pot_key`) and on every
# module constant read at trace time, so a monkeypatched one gets a new program.
_JIT_CACHE: dict = {}
_TRACE_MODULES = (limits_mod, _staeckel_prolate, _utils)


def _pot_key(pot):
    """Hashable key of a potential: its type, tree structure and leaf values."""
    leaves, treedef = jax.tree.flatten(pot)
    vals = tuple((a.dtype.str, a.shape, a.tobytes()) for a in map(np.asarray, leaves))
    return type(pot), treedef, vals


def _cached_jit(key, make):
    consts = tuple(
        (m.__name__, k, v)
        for m in _TRACE_MODULES
        for k, v in sorted(vars(m).items())
        if k.isupper()
    )
    key = (key, consts)
    if key not in _JIT_CACHE:
        _JIT_CACHE[key] = jax.jit(make())
    return _JIT_CACHE[key]


def _assert_fwd_rev(Jf, Jr, rows=None, lanes=None):
    """Finite, and forward == reverse on the outputs ``rows`` of ``lanes``.

    The same linear map: equal to roundoff (1e-10: c4 is a 4-deep nested
    jvp), floored at 1e-12 of each output's largest derivative over all
    points (an entry can be ~0 at one point).
    """
    assert np.all(np.isfinite(Jf)), np.argwhere(~np.isfinite(Jf))
    assert np.all(np.isfinite(Jr)), np.argwhere(~np.isfinite(Jr))
    rows = slice(None) if rows is None else rows
    lanes = slice(None) if lanes is None else lanes
    Jf_, Jr_ = Jf[lanes][:, rows], Jr[lanes][:, rows]
    scale = np.max(np.abs(Jf_), axis=(0, 2))[:, None]
    tol = 1e-10 * np.abs(Jf_) + 1e-12 * scale + 1e-300
    np.testing.assert_array_less(np.abs(Jf_ - Jr_), tol)


def _jac_both(f, w6s, *args, rows=None, lanes=None, rev_aux=False):
    """(jacfwd, jacrev, aux) of ``f(w6, *args) -> (out, aux)``, jitted, vmapped.

    ``args`` are per point (vmapped too); derivatives are w.r.t. ``w6``; the
    aux values come from the jacfwd compile (no separate value compile), and
    with ``rev_aux`` are ``(jacfwd aux, jacrev aux)``. Each executable is
    dropped once used, so only one is alive at a time (memory).

    Both Jacobians must be finite everywhere and agree on the outputs
    ``rows`` of the points ``lanes``; returns them for further checks.
    """
    Jf, aux = jax.jit(jax.vmap(jax.jacfwd(f, has_aux=True)))(w6s, *args)
    Jf, aux = np.asarray(Jf), jax.tree.map(np.asarray, aux)
    jax.clear_caches()  # one large executable alive at a time
    Jr, aux_r = jax.jit(jax.vmap(jax.jacrev(f, has_aux=True)))(w6s, *args)
    Jr, aux_r = np.asarray(Jr), jax.tree.map(np.asarray, aux_r)
    jax.clear_caches()
    _assert_fwd_rev(Jf, Jr, rows, lanes)
    aux = (aux, aux_r) if rev_aux else aux
    return Jf, Jr, jax.tree.map(np.asarray, aux)


def _limits_fn(pot, phi_inf="default"):
    """Jitted, vmapped (w6, D2) -> Limits through the frozen pipeline."""

    def make():
        phi = _phi(pot)

        def one(w6, d2):
            iom = integrals_of_motion(phi, w6, d2)
            F = make_F(phi, iom)
            pinf = default_phi_inf(phi, iom) if phi_inf == "default" else phi_inf
            return find_limits(F, iom, jnp.asarray(pinf))

        return jax.vmap(one)

    return _cached_jit(("limits", _pot_key(pot), repr(phi_inf)), make)


def _assert_lam_between(pot, w6s, D2s, lim):
    """Invariant: lam_min <= lam <= lam_max wherever the limits are finite."""
    lam = jax.vmap(lambda w, d: integrals_of_motion(_phi(pot), w, d).lam)(w6s, D2s)
    lam, lo, hi = (np.asarray(x) for x in (lam, lim.lam_min, lim.lam_max))
    ok = np.isfinite(lo) & np.isfinite(hi)
    assert np.all(lo[ok] <= lam[ok]), np.flatnonzero(ok & ~(lo <= lam))
    assert np.all(lam[ok] <= hi[ok]), np.flatnonzero(ok & ~(lam <= hi))


def _assert_limits_close(lim, ref_lmin, ref_lmax, ref_nu, D2s):
    """Solver tolerances: lam 3e-11 relative; nu 3e-10 of max(nu_max, 1e-4 D2)."""
    np.testing.assert_allclose(lim.lam_min, ref_lmin, rtol=3e-11, atol=0)
    np.testing.assert_allclose(lim.lam_max, ref_lmax, rtol=3e-11, atol=0)
    nu_err = np.abs(np.asarray(lim.nu_max) - ref_nu)
    nu_tol = 3e-10 * np.maximum(np.abs(ref_nu), 1e-4 * np.asarray(D2s))
    np.testing.assert_array_equal(np.isnan(lim.nu_max), np.isnan(ref_nu))
    ok = ~np.isnan(ref_nu)
    np.testing.assert_array_less(nu_err[ok], nu_tol[ok] + 1e-300)


def _D2_pipeline(pot, w6s):
    """Per-point Delta^2 with the pipeline's fallback (1e-4 r) where Delta is bad."""
    q = w6s[:, :3]
    d = _delta(pot, q, jnp.zeros(len(w6s)))
    r = jnp.linalg.norm(q, axis=-1)
    d = jnp.where(d > 0, d, 1e-4 * r)
    return d * d


def _sg(tree):
    return jax.tree.map(jax.lax.stop_gradient, tree)


def _live_limits(pot, w6, d2, *, shell=True):
    """(limits with tangents, live Fset): frozen solve on a stop-gradient pot."""
    phi, phiz = _phi(pot), _phi(_sg(pot))
    iom = integrals_of_motion(phi, w6, d2)
    iomz = integrals_of_motion(phiz, jax.lax.stop_gradient(w6), d2)
    F, Fz = make_F(phi, iom), make_F(phiz, iomz)
    lim = find_limits(Fz, iomz, default_phi_inf(phiz, iomz))
    return attach_tangents(lim, F, Fz, shell=shell), F


def _assert_jac_close(J, fd, rtol, atol_rel):
    """|J - fd| <= rtol |fd| + atol_rel * (max |fd| of the row)."""
    J, fd = np.atleast_2d(J), np.atleast_2d(fd)
    tol = rtol * np.abs(fd) + atol_rel * np.max(np.abs(fd), axis=1, keepdims=True)
    excess = np.abs(J - fd) / tol
    assert np.all(excess <= 1), (np.max(excess), J, fd)


def _scales(theta):
    """FD floor per input: |q| for positions, |v| for velocities, else |theta|."""
    th = np.asarray(theta)
    q, v = np.linalg.norm(th[:3]), np.linalg.norm(th[3:6])
    return np.concat([np.full(3, q), np.full(3, v), np.abs(th[6:])])


CIRC_R = np.arange(0.5, 16.0, 0.05)  # 310 radii: one large batch (see below)


def _near_circular_w6s(pot, e, R=CIRC_R, draw=0):
    """Planar orbits at R with epicyclic amplitude e (|dv| = e v_c), random phase.

    e = 0 is exactly circular. The phases are the ``draw``-th block of
    ``default_rng(1)``, one block per set. F(lam) there is roundoff, and
    its sign (which gala's ``F(lam_pos) < 0`` snap relied on) changes with
    the batch shape, so test both large batches and single points.
    """
    phi = _phi(pot)
    vc = np.asarray(
        jax.vmap(lambda r: jnp.sqrt(r * jax.grad(lambda q: phi(q, 0.0 * q))(r)))(
            jnp.asarray(R)
        )
    )
    ang = np.random.default_rng(1).uniform(0, 2 * np.pi, (draw + 1, len(R)))[draw]
    z = 0 * R
    vR, vphi = e * vc * np.cos(ang), vc * (1 + e * np.sin(ang))
    return jnp.asarray(np.stack([R, z, z, vR, vphi, z], 1))


def _lam_roots(pot, w6s, D2s, span=2e-2):
    """Bisection (lam_min, lam_max) of near-circular lambda shells, in numpy.

    Splits at F's maximum (F' = 0 within ``span`` lam of lam); a double root
    (F <= 0 there) returns the maximum for both.
    """
    phi = _phi(pot)

    def on_F(name):  # tau (per row) -> F.<name>(tau), all rows at once
        def make():
            def one(w, d, t):
                return getattr(make_F(phi, integrals_of_motion(phi, w, d)), name)(t)

            return jax.vmap(one)

        f = _cached_jit(("F", _pot_key(pot), name), make)
        return lambda t: np.asarray(f(w6s, D2s, jnp.asarray(t)))

    lam = jax.vmap(lambda w, d: integrals_of_motion(phi, w, d).lam)(w6s, D2s)
    lam = np.asarray(lam)
    ts = bisect(on_F("dF_lam"), (1 - span) * lam, (1 + span) * lam)
    dbl = ~(on_F("F_lam")(ts) > 0)
    lo = bisect(on_F("F_lam"), (1 - span) * lam, ts)
    hi = bisect(on_F("F_lam"), ts, (1 + span) * lam)
    return np.where(dbl, ts, lo), np.where(dbl, ts, hi)


_ROWS = ("lam_min", "lam_max", "nu_max", "m", "c2", "c3", "c4", "F2", "F4")
_PARTS = {
    "all": _ROWS,
    "limits": _ROWS[:3],
    "taylor": _ROWS[3:],
    "taylor_l": _ROWS[3:7],  # the lambda nest only (m, c2, c3, c4)
    "taylor_n": _ROWS[7:],  # the nu nest only (F2, F4)
}


def check_grad_safety(name, *, circular=False, part="all"):
    """Limits and Taylor coefficients at special and invalid points.

    At fixed Delta: generic; on the axis inside and outside the foci, and
    moving along it; a focal point; the origin; the lambda edge (z = 0,
    v_R = 0); the midplane with v_z != 0 (sigma = 0) and without (planar);
    unbound; near-planar, collapsed to nu_max = 0 below D2 MINIMUM_RANGE
    (its true tangent is not 0, but the pinned value carries none). Invalid,
    all three limits NaN: the origin with the pipeline Delta (D2 = 0),
    D2 < 0, D2 NaN, a NaN phase point, a subnormal D2, a D2 lost in
    roundoff against R^2 + z^2 (1e-300 at r ~ 5), and r ~ 1e80 (F's squared
    terms overflow).

    jacfwd and jacrev are finite and equal; invalid and
    unbound lanes leak no gradient; a generic point's limits all carry a
    tangent, a planar nu_max none. On a cuspy potential the nu-line of an
    R = 0 lane runs through the cusp, so F2, F4 there are huge-but-finite cusp
    derivatives (NFW focal point ~1e43, ~1e103): finite, but not compared.

    With ``circular``, the same two programs also take one batch of 310
    exactly circular orbits (pipeline Delta; see `check_circular_fwd_rev`).
    ``part`` differentiates the limits, the Taylor coefficients, or "all"
    in one program (a smaller program compiles in less memory).
    """
    names = _PARTS[part]

    def row(*ks):
        return [names.index(k) for k in ks if k in names]

    pot = _GRAD_POTS[name]()
    nan, Dl = np.nan, float(np.sqrt(D2))
    origin = [0.0, 0.0, 0.0, 0.05, 0.02, 0.03]
    d2_origin = float(_D2_pipeline(pot, jnp.asarray([origin]))[0])
    assert d2_origin == 0
    gen = [5.0, 0.0, 1.0, 0.05, 0.2, 0.03]
    rows = [  # (w6, D2, kind)
        (_w6s(1)[0].tolist(), D2, "generic"),
        (MID_W6.tolist(), D2, "generic"),
        (AXIS_W6.tolist(), D2, "bound"),  # R = 0 inside the foci, L_z = 0
        ([0.0, 0.0, 5.0, 0.05, 0.01, 0.02], D2, "bound"),  # outside the foci
        ([0.0, 0.0, 1.0, 0.0, 0.0, 0.02], D2, "bound"),  # along the axis
        ([0.0, 0.0, Dl, 0.05, 0.01, 0.02], D2, "bound"),  # focal point
        (origin, D2, "bound"),
        ([3.0, 0.0, 0.0, 0.0, 0.12, 0.05], D2, "bound"),  # lambda edge
        ([5.0, 0.0, 0.0, 0.05, 0.2, 0.0], D2, "planar"),
        ([3.0, 1.0, 0.0, 0.0, 0.18, 0.0], D2, "planar"),  # y != 0
        ([5.0, 0.0, 1e-6, 0.05, 0.2, 1e-7], D2, "planar"),  # collapsed to 0
        ([5.0, 0.0, 1.0, 2.0, 2.0, 2.0], D2, "unbound"),
        (origin, d2_origin, "invalid"),
        (gen, -1.0, "invalid"),
        (gen, nan, "invalid"),
        ([nan, 0.0, 1.0, 0.05, 0.2, 0.03], D2, "invalid"),
        ([5.0, 0.0, 1.0, 0.05, nan, 0.03], D2, "invalid"),
        (gen, 5e-324, "invalid"),
        (gen, 1e-300, "invalid"),
        ([1e80, 0.0, 1e80, 0.05, 0.2, 0.03], D2, "invalid"),
    ]
    w6s = jnp.asarray([w for w, _, _ in rows])
    d2s = jnp.asarray([d for _, d, _ in rows])
    kind = np.array([k for _, _, k in rows])
    n = len(rows)
    if circular:  # one compile serves both checks (the batch size is free)
        cw6s = _near_circular_w6s(pot, 0.0)
        w6s = jnp.concat([w6s, cw6s])
        d2s = jnp.concat([d2s, _D2_pipeline(pot, cw6s)])

    def f(w, d2):
        lim, F = _live_limits(pot, w, d2)
        vals = lim._asdict()
        if part != "limits":
            vals |= taylor_coeffs(F, lim)._asdict()
        out = jnp.stack([vals[k] for k in names])
        return jnp.where(jnp.isfinite(out), out, 0.0), (out, lim)

    cusp = name in ("nfw", "mw2022", "composite")
    Jf, Jr, ((out, lim), (_, lim_r)) = _jac_both(
        f,
        w6s,
        d2s,
        rows=row(*_ROWS[:7]) if cusp else None,
        lanes=slice(0, n),
        rev_aux=True,
    )
    if circular:
        cl = slice(n, None)
        lf, lr = (jax.tree.map(lambda x: x[cl], x) for x in (lim, lim_r))
        sel = row("lam_min", "lam_max", "nu_max", "c2", "c3", "c4")
        if sel:
            check_circular_fwd_rev(
                Jf[cl][:, sel], Jr[cl][:, sel], out[cl][:, sel], lf, lr
            )
    Jf, Jr, out, planar = Jf[:n], Jr[:n], out[:n], lim.planar[:n]
    L3 = np.stack([lim.lam_min[:n], lim.lam_max[:n], lim.nu_max[:n]], 1)
    bound = np.isin(kind, ["generic", "bound", "planar"])
    assert np.all(np.isfinite(L3[bound]))
    assert np.all(np.isnan(L3[kind == "invalid"]))  # nu_max too
    assert np.all(np.isnan(L3[kind == "unbound"][:, 1]))
    assert np.all(planar[kind == "planar"])
    for J in (Jf, Jr):  # invalid lanes: no gradient at all
        np.testing.assert_array_equal(J[kind == "invalid"], 0.0)
    # unbound: lam_max (and m) NaN, and no gradient leaks from them
    leak = row("lam_max", "m", "c2", "c3", "c4")
    np.testing.assert_array_equal(Jf[kind == "unbound"][:, leak], 0.0)
    lims = row("lam_min", "lam_max", "nu_max")
    if lims:
        assert np.all(np.any(Jf[kind == "generic"][:, lims] != 0, axis=-1))
        np.testing.assert_array_equal(Jf[kind == "planar"][:, lims[2]], 0.0)


def check_circular_fwd_rev(Jf, Jr, out, lf, lr):
    """Exactly circular orbits, one batch of 310 radii, pipeline Delta.

    Every lane is finite, and nearly all are snapped shells, with the same
    decisions in the jacfwd and the jacrev program (gala's F(lam_pos) < 0
    clause read a roundoff-level sign that differs between the two
    compiles), and their Jacobians agree. ``out``, ``Jf``, ``Jr`` are
    `check_grad_safety`'s rows among (lam_min, lam_max, nu_max, c2, c3, c4).
    """
    for lim in (lf, lr):
        assert np.mean(lim.snapped) > 0.99
        for k in ("lam_min", "lam_max", "nu_max"):
            assert np.all(np.isfinite(getattr(lim, k))), k
    for k in ("snapped", "shell"):  # the same decisions in both programs
        np.testing.assert_array_equal(getattr(lf, k), getattr(lr, k), err_msg=k)
    for k in ("lam_min", "lam_max"):  # and the same values, to roundoff
        np.testing.assert_allclose(getattr(lf, k), getattr(lr, k), rtol=1e-13)
    assert np.all(np.isfinite(out))  # so unmasked: Jf, Jr are out's
    scale = np.max(np.abs(Jf), axis=2, keepdims=True) + 1e-300  # per lane, output
    # measured <= 1.1e-8 here and 2.5e-8 at e = 1e-8 (lanes near the snap
    # width); 1e-7 leaves room for platform round-off
    np.testing.assert_array_less(np.abs(Jf - Jr) / scale, 1e-7)


def check_dF_limit_sites(name):
    """F and F' at the tau = D2 and u = 0 limit sites and at the phase point.

    At z = 0 the lambda-side F'(D2) limit term sits at Phi(0, sigma = 0), the
    origin, unless kept off the axis: both modes finite, equal.
    """
    phi = _phi(_GRAD_POTS[name]())

    def f(w):
        iom = integrals_of_motion(phi, w, D2)
        F = make_F(phi, iom)
        d2 = jnp.asarray(D2)
        out = jnp.stack(
            [
                F.dF_lam(d2),
                F.dF_nu(d2),
                F.dF_nu(jnp.asarray(0.0)),
                F.F_lam(d2),
                F.F_nu(jnp.asarray(0.0)),
                F.F_lam(iom.lam),
                F.F_nu(iom.absnu),
                F.dF_lam(iom.lam),
            ]
        )
        return out, out

    w6s = jnp.stack([_w6s(1)[0], AXIS_W6, MID_W6])
    _, _, vals = _jac_both(f, w6s)
    assert np.all(np.isfinite(vals))


# ---------------------------------------------------------------------------
# Shared by the tangent and Taylor tests: one jacfwd and one jacrev compile.


def _param_out(theta, *, plain):
    """Limits and Taylor coefficients w.r.t. (w6, MN parameters).

    (lam_min, lam_max, nu_max, c2, c3, c4, F2, F4); with ``plain`` also the
    coefficients from two separate nested-jvp nests (not `taylor_coeffs`'
    fused one). Aux: (the outputs, the Limits).
    """
    lim, F = _live_limits(_mn_p(theta[6:]), theta[:6], MN_D2)
    t = taylor_coeffs(F, lim)
    rows = [lim.lam_min, lim.lam_max, lim.nu_max, t.c2, t.c3, t.c4, t.F2, t.F4]
    if plain:
        m = 0.5 * (lim.lam_min + lim.lam_max)
        _, _, d2, d3, d4 = limits_mod._derivs(F.F_lam, m, 4)
        _, _, u2, _, u4 = limits_mod._derivs(F.F_nu_u, jnp.zeros_like(m), 4)
        rows += [d2 / 2, d3 / 6, d4 / 24, u2 / 2, u4 / 24]
    out = jnp.stack(rows)
    return out, (out, lim)


@ft.cache
def _param_jacs():
    """Compute jacfwd (with the plain nests) and jacrev of `_param_out`, two points.

    A generic point and the lambda edge (z = 0, v_R = 0: one limit is the
    phase point's lambda). One compile per mode serves the tangent and the
    Taylor-autodiff tests (cached per process); the values come from the
    jacfwd program.
    """
    w6s = jnp.asarray(
        [
            [8.0, 0.5, 0.6, 0.03, 0.2, 0.04],  # generic
            [3.0, 0.0, 0.0, 0.0, 0.18, 0.05],  # z = 0, v_R = 0: lambda edge
        ]
    )
    thetas = jnp.concat([w6s, jnp.broadcast_to(MN_P, (2, 3))], axis=1)
    fwd = jax.jacfwd(lambda t: _param_out(t, plain=True), has_aux=True)
    rev = jax.jacrev(lambda t: _param_out(t, plain=False), has_aux=True)
    Jf, (vals, lim0) = jax.jit(jax.vmap(fwd))(thetas)
    Jr, _ = jax.jit(jax.vmap(rev))(thetas)
    return w6s, thetas, np.asarray(Jf), np.asarray(Jr), np.asarray(vals), lim0

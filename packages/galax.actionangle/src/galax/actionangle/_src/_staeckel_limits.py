"""Staeckel-fudge turning points (per point, then ``vmap``).

The fused turning-point solve, its implicit-function tangents, and the
Taylor coefficients of the Q-form regimes.
"""

__all__: list[str] = []

from collections.abc import Callable
from jaxtyping import Array
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._staeckel_prolate import (
    IOM,
    LOG_TAU_MAX,
    Fset,
    _F_semi_val,
    _F_val,
    _semi_tau,
    integrals_of_motion,
    make_F,
)
from ._utils import chandrupatla, tang

# ---------------------------------------------------------------------------
# Turning points. Values only, on frozen inputs; the
# implicit-function tangents are attached separately.

N_ROOT_ITER = 16  # fused Chandrupatla iterations
MINIMUM_RANGE = 1e-10  # C++: smallest nu / lambda interval treated as non-zero
FIRST_STEP_EPS = 1e-3  # clip of the quadratic first step, as a bracket fraction
SNAP_NOISE = 8.0  # snap a double root within this factor of F's roundoff
LADDER = 10.0  # geometric ratio of the outer-end ladders (see find_limits)
N_LADDER_LO, N_LADDER_HI = 2, 7  # ladder points below and above the vertex
N_LADDER_D2 = 3  # ladder points towards lam_lower (see find_limits)
LADDER_FAR = (8.0, 24.0, 72.0)  # log(tau / v) of the far ladder points above v


class Limits(NamedTuple):
    """Turning points and how each was found (see `find_limits`)."""

    lam_min: Array
    lam_max: Array
    nu_max: Array
    shell: Array  # lam_max - lam_min < D2 MINIMUM_RANGE: both set to lam
    planar: Array  # nu_max < D2 MINIMUM_RANGE: set to 0
    nu_pinned: Array  # L_z = 0 box/tube decision set nu_max to 0 or D2
    lmin_pinned: Array  # lam_min set to D2 by a decision, not a root
    snapped: Array  # double root (see find_limits): lam_min = lam_max = lam
    # The origin of the lam_max lane's variable (F > 0 there), not the phase
    # point's lambda; lam where that lane is not solved (edge, snapped).
    lam_pos: Array


def _lane_terms(Fz: Fset, D2: Array, tau: Array) -> tuple[Array, Array, Array]:
    """(Phi, mult, I) of the C++ F_tau at ``tau``, with one ``phi`` call.

    tau = D2 is evaluated at 2 D2: F's Phi term carries a factor tau - D2 = 0
    there, so the value is unchanged and Phi never sees a focal point.
    """
    R, z, mult, I = Fz.lane_coords(jnp.where(tau == D2, 2 * D2, tau))
    return Fz.phi(R, z), mult, I


def _quad_roots(f: Array, df: Array, d2f: Array) -> tuple[Array, Array]:
    """Offsets (below, above) of the roots of f + f' x + f'' x^2 / 2, f'' < 0.

    Cancellation-free: q / (f''/2) and f / q, q = -(f' + sgn(f') sqrt(disc))/2,
    disc = f'^2 - 2 f f''. At a turning point (f ~ 0) the far root is the
    first; 2f / (-f' -+ sqrt(disc)) would be 0/0 there. NaN or inf when
    disc < 0 or q = 0.
    """
    disc = df * df - 2 * f * d2f
    sq = jnp.sqrt(jnp.where(disc >= 0, disc, jnp.nan))
    q = -0.5 * (df + jnp.where(df >= 0, sq, -sq))
    r1, r2 = q / (0.5 * d2f), f / q
    return jnp.minimum(r1, r2), jnp.maximum(r1, r2)


def _noise_snap(
    f: Array, df: Array, d2f: Array, dlam: Array, S: Array, F0: Array
) -> Array:
    """Double root within F's roundoff: snap lam_min = lam_max = lam.

    F = (E tau - I - Phi mult)(tau - D2) - L_z^2 tau / 2 is computed from
    terms of size S = |E lam| + |I| + |Phi mult| and F0 = L_z^2 lam / 2, so
    near lam its roundoff is ~ eps (S (|lam - D2| + |x|) + F0) at an offset
    x. The quadratic model's far root lies h = (sqrt(disc) + |f'|) / |f''|
    from lam; where |f''| h^2 < SNAP_NOISE eps (S (|lam - D2| + h) + F0),
    F's maximum is within SNAP_NOISE of its roundoff and the root pair is
    unresolvable. h is also the error of the snap. ``dlam`` is lam - D2.
    Positive form, so a NaN f'' (at lam = D2) does not snap.
    """
    disc = df * df - 2 * f * d2f
    h = (jnp.sqrt(jnp.where(disc > 0, disc, 0.0)) + jnp.abs(df)) / jnp.abs(d2f)
    eps = jnp.finfo(jnp.result_type(f)).eps  # type: ignore[no-untyped-call]
    noise = eps * (S * (jnp.abs(dlam) + h) + F0)
    return (d2f < 0) & (jnp.abs(d2f) * h * h < SNAP_NOISE * noise)


def _first_step(
    lam: Array,
    f: Array,
    df: Array,
    d2f: Array,
    a: Array | float,
    b: Array | float,
    *,
    sign: float,
    lam_pos: Array | None = None,
) -> Array:
    """First trial point of a lambda lane from the quadratic model of F at lam.

    tau* = lam + x, x the model's root below lam (sign -1, lam_min) or above
    it (sign +1, lam_max), see `_quad_roots`. With ``lam_pos`` (the lam_max
    lane) it is mapped to s* = l/(1+l), l = log(tau*/lam_pos). It is accepted
    only if ``disc >= 0 & f'' < 0 & a < x* < b`` (so any NaN rejects it) and is
    then clipped to FIRST_STEP_EPS (b - a) inside the bracket; otherwise the
    midpoint is used.
    """
    lo, hi = _quad_roots(f, df, d2f)
    x = lam + (lo if sign < 0 else hi)
    if lam_pos is not None:
        ell = jnp.log(x / lam_pos)
        x = ell / (1 + ell)
    accept = (d2f < 0) & (a < x) & (x < b)
    c_lo, c_hi = a + FIRST_STEP_EPS * (b - a), b - FIRST_STEP_EPS * (b - a)
    x = jnp.where(x < c_lo, c_lo, jnp.where(x > c_hi, c_hi, x))
    return jnp.where(accept, x, 0.5 * (a + b))


def default_phi_inf(phi: Callable[[Array, Array], Array], iomz: IOM) -> Array:
    """Phi at the lambda_max lane's guard point, log tau = 230, on nu = 0."""
    R = jnp.sqrt(jnp.exp(LOG_TAU_MAX) - iomz.D2)
    return phi(R, jnp.zeros_like(R))


def _ladder_side(t: Array, f: Array) -> tuple[Array, Array, Array, Array, Array]:
    """Bracket on one side of the vertex from the ladder ``t`` (t[0] = v).

    The leading run of F > 0 points ends at the inner end (F > 0, nearest the
    root); the next point, if F < 0 there, is the outer end. Returns (inner,
    f(inner), outer, f(outer), closed). NaN points (outside the branch) end
    the run and never close it.
    """
    n = jnp.sum(jnp.cumprod((f > 0).astype(int)))  # >= 1: f[0] = F(v) > 0 where used
    i_in = jnp.maximum(n - 1, 0)
    i_out = jnp.minimum(n, len(t) - 1)
    closed = (n < len(t)) & (f[i_out] < 0)
    return t[i_in], f[i_in], t[i_out], f[i_out], closed


def find_limits(Fz: Fset, iomz: IOM, phi_inf: Array) -> Limits:
    """lam_min, lam_max, nu_max: a branchless port of the C++ ``find_limits``.

    ``Fz`` and ``iomz`` must be frozen (built from stop-gradient inputs and a
    stop-gradient potential); the outputs carry no gradient. The three roots
    are found by one fused Chandrupatla loop:

    - nu_max: log tau in [log |nu|, log nu_upper] (|nu| floored at D2
      MINIMUM_RANGE^2); |nu| if F(0) <= 0 or no bracket.
    - lam_min: log tau in [log a_lam, log b_lam].
    - lam_max: s in [0, b_s], tau = lam_s exp(s/(1-s)), on F/tau^2, and
      E - phi_inf past the guard.

    By default b_lam = lam_s is the nudged lambda, a_lam = lam_lower and
    b_s = 1. Where F's quadratic model at lambda has a maximum v with
    F(v) > 0, the brackets are split at v and tightened along ladders of
    points either side; the lam_max bracket is also closed from far points
    when unsplit (see the comments in the body). lam_s is returned as
    ``lam_pos``.

    Snapped (a double root, lam_min = lam_max = lam): at the lambda edge
    where the quadratic model of F at lam has no real root (gala's test), or
    on any lane where the model's root pair lies within F's own roundoff
    (`_noise_snap`). This replaces gala's ``F(lam_pos) < 0`` clause, which
    reads the sign of a roundoff-level F and so depends on how the batch is
    compiled.

    A lambda lane without a valid bracket returns NaN (the point is then
    masked); unlike the C++, the MINIMUM_RANGE collapse keeps that NaN. An
    invalid point (``iomz.valid`` False) or a non-finite E + I3 + L_z (C++:
    ``isfinite(E + I3 + Lz)``) returns NaN for all three limits, flags False.
    """
    io = iomz
    D2, lam, absnu, E = io.D2, io.lam, io.absnu, io.E
    # L_z^2, not L_z, decides L_z = 0: an L_z that squares to 0 (|L_z| below
    # ~1e-154) is 0 in F too, and F(D2) = 0 then needs the L_z = 0 logic.
    Lz2 = io.Lz**2
    FD2 = -Lz2 * D2 / 2  # F(D2), on either branch
    Dm, Dp = D2 * (1 - 1e-15), D2 * (1 + 1e-15)

    # F at 0, |nu|, D2 (1 -+ 1e-15) and lam: one phi call.
    t5 = jnp.stack([jnp.zeros_like(D2), absnu, Dm, Dp, lam])
    Phi5, mult5, I5 = _lane_terms(Fz, D2, t5)
    f0, fnu, fm, fp, fl = _F_val(io, t5, Phi5, mult5, I5)
    dfl = Fz.dF_lam(lam)

    # L_z = 0: F(D2) = 0, so box vs tube is decided just either side of D2.
    Lz0 = Lz2 == 0
    nu_pinned = Lz0 & ~((fm < 0) & (f0 > 0))
    nu_upper, fnu_upper = jnp.where(Lz0, Dm, D2), jnp.where(Lz0, fm, FD2)
    lam_lower, flam_lower = jnp.where(Lz0, Dp, D2), jnp.where(Lz0, fp, FD2)

    # The lambda edge: F(lam) <= 0 or a tiny Newton step. Assign the limit
    # on the side F' points away from, and nudge lam_pos into F > 0.
    # F' = 0 (a flat F, e.g. a maximum below 0 by roundoff): no Newton step,
    # so lam_pos stays finite (it is exported). Double where: 0/0 otherwise.
    flat = dfl == 0
    dx = jnp.where(flat, 0.0, -fl / jnp.where(flat, 1.0, dfl))
    edge = (fl <= 0) | (~flat & (jnp.abs(dx) < lam * MINIMUM_RANGE))
    lmin_edge, lmax_edge = edge & (dfl >= 0), edge & (dfl <= 0)
    # The nudge is the larger of the Newton step and min(half the vertex
    # offset, lam MINIMUM_RANGE). F''(lam) is 0/0 = NaN at lam = D2 (R = 0
    # inside the foci): the nudge is lam MINIMUM_RANGE there, and the model's
    # F'' (below) is taken at the nudged point instead.
    sgn = jnp.where(dfl >= 0, 1.0, -1.0)
    at_D2 = ~(lam > D2)
    d2f = Fz.d2F_lam(
        jnp.where(at_D2, lam + jnp.fmax(dx, lam * MINIMUM_RANGE) * sgn, lam)
    )
    off = jnp.where(
        at_D2,
        lam * MINIMUM_RANGE,
        jnp.fmin(0.5 * jnp.abs(dfl / d2f), lam * MINIMUM_RANGE),
    )
    lam_pos = jnp.where(edge, lam + jnp.fmax(dx, off) * sgn, lam)

    # A bracket whose end |F| dwarfs F near the root makes interpolation
    # creep: near a lambda turning point (lam_pos is almost the other root),
    # near the axis inside the foci, at a cusp, and wherever a root is
    # decades from lambda. So split the brackets at the model's vertex v
    # (F's maximum) and close each half from a ladder of points:
    # - above v: v + (o - v) LADDER^k, o = 2r - v (the model's root r
    #   reflected about v; k = 0 is o itself), and the far points lam_pos
    #   exp(LADDER_FAR) for roots decades out;
    # - below v: v - (v - o) LADDER^k, and lam_lower + (v - lam_lower)
    #   LADDER^-k towards a root just above D2 (tiny L_z, near the axis).
    # On each side the inner end is the last point of the leading run with
    # F > 0, the outer end the next point if F < 0 there (else lam_lower, or
    # open). Unsplit, the lam_max lane is closed from lam_pos's far points.
    # F at lam_pos, v and every ladder point: one phi call.
    xl, xh = _quad_roots(fl, dfl, d2f)
    v = lam - dfl / d2f
    dl, dh = v - (2 * (lam + xl) - v), (2 * (lam + xh) - v) - v
    g_lo = LADDER ** jnp.arange(N_LADDER_LO, dtype=lam.dtype)
    g_hi = LADDER ** jnp.arange(N_LADDER_HI, dtype=lam.dtype)
    g_d2 = LADDER ** -jnp.arange(1, N_LADDER_D2 + 1, dtype=lam.dtype)
    t_lo = jnp.concat(
        [
            jnp.where(dl > 0, v - dl * g_lo, jnp.nan),
            lam_lower + (v - lam_lower) * g_d2,
        ]
    )
    t_far = lam_pos * jnp.exp(jnp.asarray(LADDER_FAR, dtype=lam.dtype))
    t_hi = jnp.concat(
        [
            jnp.where(dh > 0, v + dh * g_hi, jnp.nan),
            jnp.where(jnp.log(t_far) < LOG_TAU_MAX, t_far, jnp.nan),
        ]
    )
    tk = jnp.concat([lam_pos[None], v[None], t_lo, t_hi])
    tk = jnp.where(tk > lam_lower, tk, jnp.nan)
    fk = _F_val(io, tk, *_lane_terms(Fz, D2, tk))
    f_lp, f_v = fk[0], fk[1]
    n_lo = N_LADDER_LO + N_LADDER_D2
    f_lo, f_hi = fk[2 : 2 + n_lo], fk[2 + n_lo :]

    # Snapped: gala's near-shell test at the edge (the model has no real
    # root), or a root pair within F's roundoff anywhere (`_noise_snap`).
    # F(lam) for the snap from the momenta, (lam - |nu|)^2 lamdot^2 / (8 lam):
    # exact and >= 0, where the F above is a difference of O(1) terms whose
    # roundoff changes with the compiled batch shape.
    S = jnp.abs(E * lam) + jnp.abs(I5[4]) + jnp.abs(Phi5[4] * mult5[4])
    f_mom = (lam - absnu) ** 2 * io.lamdot**2 / (8 * lam)
    disc = dfl * dfl - 2 * fl * d2f
    snapped = (edge & (d2f < 0) & (disc < 0)) | _noise_snap(
        f_mom, dfl, d2f, lam - D2, S, Lz2 * lam / 2
    )
    lmin_pinned = (
        (Lz0 & ~(fp < 0) & ~lmin_edge) | (lmax_edge & (lam == D2))
    ) & ~snapped

    lmin_done = lmin_pinned | lmin_edge | snapped
    lmin_set = jnp.where(lmin_pinned, D2, lam)  # value of a done lam_min lane

    # The split and the ladder brackets (see above); positive forms.
    split_ok = (d2f < 0) & (f_v > f_lp) & (f_v > 0)
    order = jnp.argsort(-tk[2 : 2 + n_lo])  # descending; NaN (dropped) last
    lo_in, flo_in, lo_out, flo_out, lo_closed = _ladder_side(
        jnp.concat([v[None], tk[2 : 2 + n_lo][order]]),
        jnp.concat([f_v[None], f_lo[order]]),
    )
    order = jnp.argsort(t_hi)  # ascending; NaN (dropped) last
    hi_in, fhi_in, hi_out, fhi_out, hi_closed = _ladder_side(
        jnp.concat([v[None], t_hi[order]]), jnp.concat([f_v[None], f_hi[order]])
    )
    # unsplit: lam_pos and its far points
    n_far = len(LADDER_FAR)
    far_in, ffar_in, far_out, ffar_out, far_closed = _ladder_side(
        jnp.concat([lam_pos[None], t_hi[-n_far:]]),
        jnp.concat([f_lp[None], f_hi[-n_far:]]),
    )
    lmin_b = jnp.where(split_ok, lo_in, lam_pos)
    lmin_fb = jnp.where(split_ok, flo_in, f_lp)
    near_l = split_ok & lo_closed
    lmin_a = jnp.where(near_l, lo_out, lam_lower)
    lmin_fa = jnp.where(near_l, flo_out, flam_lower)
    split = jnp.where(split_ok, hi_in, far_in)  # the lam_max lane's origin
    f_split = jnp.where(split_ok, fhi_in, ffar_in)
    near_h = jnp.where(split_ok, hi_closed, far_closed)
    out_h, fout_h = (
        jnp.where(split_ok, hi_out, far_out),
        jnp.where(split_ok, fhi_out, ffar_out),
    )
    ell = jnp.log(jnp.where(near_h, out_h, 2 * split) / split)
    lmax_b = jnp.where(near_h, ell / (1 + ell), 1.0)
    lmax_fb = jnp.where(near_h, fout_h / out_h**2, E - phi_inf)

    # accept, clip and midpoint are evaluated in tau, then log-mapped below
    lmin_x0 = _first_step(lam, fl, dfl, d2f, lmin_a, lmin_b, sign=-1.0)

    # Lane variables: log tau, log tau, s. The lam_min lane runs in log tau:
    # with a small (fallback) Delta its bracket spans many decades, and
    # bisection in tau needs ~log2(lam/lam_min) steps before interpolation
    # can take over. So does the nu lane: its bracket [|nu|, D2] spans
    # decades near the centre, and on a cusp (Phi ~ r) F ~ a + b sqrt(tau).
    nu_done = nu_pinned | ~(f0 > 0)
    done0 = jnp.stack([nu_done, lmin_done, lmax_edge | snapped])
    # log tau of the nu lane, floored where |nu| = 0 (z = 0) at D2
    # MINIMUM_RANGE^2: a root below that is collapsed to 0 anyway.
    nu_floor = D2 * MINIMUM_RANGE**2
    log_nu = jnp.log(jnp.where(absnu > nu_floor, absnu, nu_floor))
    x0 = jnp.stack(
        [
            log_nu,
            jnp.log(jnp.where(lmin_done, lmin_set, lmin_x0)),
            _first_step(lam, fl, dfl, d2f, 0.0, lmax_b, sign=1.0, lam_pos=split),
        ]
    )
    a = jnp.stack([log_nu, jnp.log(lmin_a), jnp.zeros_like(lam)])
    b = jnp.stack([jnp.log(nu_upper), jnp.log(lmin_b), lmax_b])
    fa = jnp.stack([fnu, lmin_fa, f_split / split**2])
    fb = jnp.stack([fnu_upper, lmin_fb, lmax_fb])

    def lanes(x: Array) -> Array:
        ts, beyond, ok = _semi_tau(x[2], split)
        tau = jnp.stack([jnp.exp(x[0]), jnp.exp(x[1]), ts])
        Phi, mult, I = _lane_terms(Fz, D2, tau)
        F = _F_val(io, tau[:2], Phi[:2], mult[:2], I[:2])
        Fs = _F_semi_val(io, ts, Phi[2], mult[2], I[2])
        Fs = jnp.where(ok, jnp.where(beyond, E - phi_inf, Fs), jnp.nan)
        return jnp.concat([F, Fs[None]])

    r = chandrupatla(lanes, a, b, fa, fb, x0, done0, N_ROOT_ITER)

    # exp(log tau) can round an endpoint root an ulp outside the bracket:
    # clamp back into it. NaN (no bracket): |nu| (C++ fallback).
    nu_tau = jnp.exp(r[0])
    nu_tau = jnp.where(
        nu_tau < absnu, absnu, jnp.where(nu_tau > nu_upper, nu_upper, nu_tau)
    )
    nu_max = jnp.where(
        nu_pinned,
        jnp.where(fm < 0, 0.0, D2),
        jnp.where(~nu_done & jnp.isfinite(nu_tau), nu_tau, absnu),
    )
    # exp(log tau) can round an endpoint root to an ulp outside the bracket
    # (below D2 for tiny L_z): clamp back into the bracket [lmin_a, lmin_b].
    lmin_tau = jnp.exp(r[1])
    lmin_tau = jnp.where(
        lmin_tau < lmin_a, lmin_a, jnp.where(lmin_tau > lmin_b, lmin_b, lmin_tau)
    )
    lam_min = jnp.where(lmin_done, lmin_set, lmin_tau)
    ts, beyond, ok = _semi_tau(r[2], split)
    lam_max = jnp.where(done0[2], lam, jnp.where(ok & ~beyond, ts, jnp.nan))

    # ignore extremely small intervals (positive form for shell: NaN stays NaN)
    ok = io.valid & jnp.isfinite(E + io.I3 + io.Lz)
    planar = ok & ~(nu_max >= D2 * MINIMUM_RANGE)
    shell = ok & (lam_max - lam_min < D2 * MINIMUM_RANGE)
    lam_min = jnp.where(shell, lam, lam_min)
    lam_max = jnp.where(shell, lam, lam_max)
    nu_max = jnp.where(planar, 0.0, nu_max)
    nan = jnp.nan
    return jax.lax.stop_gradient(
        Limits(
            lam_min=jnp.where(ok, lam_min, nan),
            lam_max=jnp.where(ok, lam_max, nan),
            nu_max=jnp.where(ok, nu_max, nan),
            shell=shell,
            planar=planar,
            nu_pinned=ok & nu_pinned,
            lmin_pinned=ok & lmin_pinned,
            snapped=ok & snapped,
            lam_pos=jnp.where(ok, jnp.where(done0[2], lam, split), nan),
        )
    )


# ---------------------------------------------------------------------------
# Tangents: the solver's values with the implicit-function
# tangent reattached, `L = sg(L0) - tang(F(sg L0)) / sg(F'(L0))`.


def _tang_at(Fn: Callable[[Array], Array], L0: Array, D2: Array, safe: Array) -> Array:
    """tang(Fn(L0)): the theta-derivative of F at fixed tau = L0.

    Zero (and NaN-free) at non-finite L0 and at L0 = D2, where F = -L_z^2 D2/2;
    the branch closures are never evaluated there, so never at a focal point.
    L0 = D2 is reached with L_z = 0 (R = 0, or a pinned limit), and also
    with a tiny L_z near the axis (nu_max rounds to D2): the zeroed tangent
    is then O(L_z) (measured ~3e-9 relative).
    """
    ok = jnp.isfinite(L0) & (L0 != D2)
    return jnp.where(ok, tang(Fn(jnp.where(ok, L0, safe))), 0.0)


def _over(num: Array, den: Array) -> Array:
    """Num / den, and 0 (NaN-free) where den is 0 or non-finite."""
    ok = jnp.isfinite(den) & (den != 0)
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)


def attach_tangents(lim: Limits, F: Fset, Fz: Fset) -> Limits:
    """Reattach implicit-function tangents to `find_limits`'s values.

    ``F`` is the live Fset (its tangents are the ones attached), ``Fz`` the
    frozen one (the denominators). Per limit, values unchanged:

    - simple root (solver roots, the edge's ``= lam`` and the ``|nu|``
      fallback): ``-tang(F(L0)) / F'(L0)``;
    - double root (shell, incl. snapped): ``-tang(F'(L0)) / F''(L0)``;
    - pinned (lam_min = D2; nu_max = 0 or D2): none.

    Every denominator is guarded (0 on snapped lanes, NaN at D2). First order
    only: second derivatives miss the d2F/dL dtheta and dF' terms.
    """
    D2 = F.D2
    sg = jax.lax.stop_gradient
    lmin, lmax, nu = sg(lim.lam_min), sg(lim.lam_max), sg(lim.nu_max)

    lam2 = jnp.stack([lmin, lmax])
    simple = -_over(_tang_at(F.F_lam, lam2, D2, 2 * D2), sg(Fz.dF_lam(lam2)))
    double = -_over(_tang_at(F.dF_lam, lmax, D2, 2 * D2), sg(Fz.d2F_lam(lmax)))
    t_lam = jnp.where(lim.shell, double, simple)

    t_nu = -_over(_tang_at(F.F_nu, nu, D2, 0.5 * D2), sg(Fz.dF_nu(nu)))
    t_nu = jnp.where(lim.planar | lim.nu_pinned, 0.0, t_nu)
    return lim._replace(
        lam_min=lmin + t_lam[0], lam_max=lmax + t_lam[1], nu_max=nu + t_nu
    )


# ---------------------------------------------------------------------------
# Taylor coefficients of the near-shell and near-planar Q.

NEAR_SHELL = 1e-2  # w < NEAR_SHELL hi: quartic-Taylor Q on the lambda branch
NEAR_PLANAR = 1e-4  # nu_max < NEAR_PLANAR D2: candidate for the even-Taylor nu Q
NEAR_PLANAR_SERIES = 3e-5  # ... used iff |F4| nu_max < NEAR_PLANAR_SERIES |F2|


class Taylor(NamedTuple):
    """Taylor regimes of Q and their coefficients (see `taylor_coeffs`)."""

    use_l: Array  # near-shell: Q = -(c2 + c3 x + c4 (x^2 + w^2/4)), x = tau - m
    m: Array  # (lam_min + lam_max) / 2
    c2: Array  # F^(k)(m) / k! on the lambda branch
    c3: Array
    c4: Array
    use_n: Array  # near-planar: Q = -(F2 + F4 (hi + u^2))
    F2: Array  # F(u) = F0 + F2 u^2 + F4 u^4 on the nu branch
    F4: Array


def _derivs(f: Callable[[Array], Array], x: Array, n: int) -> tuple[Array, ...]:
    """(f, f', ..., f^(n)) at x, by n nested `jax.jvp`s."""

    def nest(k: int) -> Callable[[Array], tuple[Array, ...]]:
        if k == 0:
            return lambda t: (f(t),)
        inner = nest(k - 1)

        def g(t: Array) -> tuple[Array, ...]:
            p, d = jax.jvp(inner, (t,), (jnp.ones_like(t),))
            return (*p, d[-1])

        return g

    return nest(n)(x)


def _candidates(lim: Limits, D2: Array) -> tuple[Array, Array, Array]:
    """(m, near-shell, near-planar candidate) flags from the limits alone.

    near-shell is final (use_l); near-planar is a candidate: use_n also
    needs the coefficients (see `taylor_coeffs`). No gradient.
    """
    sg = jax.lax.stop_gradient
    lo, hi, nu = lim.lam_min, lim.lam_max, sg(lim.nu_max)
    m = 0.5 * (lo + hi)
    ok = jnp.isfinite(m) & (m > D2)
    near_l = (sg(hi - lo) < NEAR_SHELL * sg(hi)) & ok  # finite c_k only
    near_n = nu < NEAR_PLANAR * sg(D2)
    return m, near_l, near_n


def taylor_coeffs(F: Fset, lim: Limits) -> Taylor:
    """Coefficients of the Q-form Taylor regimes, from nested jvps of live F.

    c2..c4 are the tau-Taylor coefficients of F_lam at m = (lam_min+lam_max)/2
    (NaN where m is not inside the lambda branch); F2, F4 the u-Taylor
    coefficients of F_nu_u at u = 0. The flags carry no gradient; use_l is
    False where the c_k are NaN (m = D2: R below ~1e-8 Delta inside the foci).

    use_n (near-planar) needs nu_max < NEAR_PLANAR D2 and a series that
    converges over [0, nu_max]: F2, F4 finite and |F4| nu_max <
    NEAR_PLANAR_SERIES |F2|, so the dropped u^6 term is ~1e-9 of Q (measured
    6 (|F4| nu_max / |F2|)^2). That keeps smooth potentials on the series
    near the axis, where the direct quadrature loses up to ~1e-4 to
    cancellation, and keeps lanes whose nu-line runs through or near a cusp
    (F2, F4 up to ~1e100 there: NFW, MW2022's halo) on the direct quadrature.
    """
    m, use_l, near_n = _candidates(lim, F.D2)
    ok = jnp.isfinite(m) & (m > F.D2)
    c = _coeffs(F, jnp.where(ok, m, 2 * F.D2))
    c2, c3, c4 = (jnp.where(ok, c[k], jnp.nan) for k in range(3))
    sg = jax.lax.stop_gradient
    F2, F4 = sg(c[3]), sg(c[4])
    use_n = (
        near_n
        & jnp.isfinite(F2)
        & jnp.isfinite(F4)
        & (jnp.abs(F4) * sg(lim.nu_max) < NEAR_PLANAR_SERIES * jnp.abs(F2))
    )
    return Taylor(use_l, m, c2, c3, c4, use_n, c[3], c[4])


def _coeffs(F: Fset, x: Array) -> Array:
    """(c2, c3, c4) of F_lam at x and (F2, F4) of F_nu_u at u = 0, stacked.

    Both jvp nests are written in this one function with one stacked
    output, and plain autodiff runs through it. Consumers must read the
    coefficients by indexing (the `Taylor` fields): XLA then fuses the two
    nests' value-and-gradient work. Summing the stacked output, or calling
    the two nests separately, compiles to ~2.6x the flops and ~4x the CPU
    under ``vmap`` (value and gradient; pinned by a flop-count test). A
    `jax.custom_jvp` wrapper gave bitwise the same reverse-mode derivatives
    and no gain, so there is none.
    """
    _, _, d2, d3, d4 = _derivs(F.F_lam, x, 4)
    _, _, u2, _, u4 = _derivs(F.F_nu_u, jnp.zeros_like(x), 4)
    return jnp.stack([d2 / 2, d3 / 6, d4 / 24, u2 / 2, u4 / 24])


def taylor_coeffs_batch(
    phi: Callable[[Array, Array], Array],
    w6: Array,
    D2: Array,
    lim: Limits,
    size: int | None = None,
) -> Taylor:
    """`taylor_coeffs` on a batch, computed only on the lanes that may use it.

    ``w6`` is (N, 6), ``D2`` and the ``lim`` fields (N,): the batch is the
    leading axis, and the live F of each lane is rebuilt here from
    (``phi``, ``w6``, ``D2``), as the pipeline builds it. The candidates
    (near-shell, or nu_max < NEAR_PLANAR D2) are gathered into ``size``
    lanes (default max(16, N // 16)), their coefficients computed, and
    scattered back; when more lanes are candidates, every lane is computed
    (one `jax.lax.cond` on the whole batch). Either way the result on a
    candidate lane is `taylor_coeffs`' value and derivative; on the other
    lanes the coefficients are NaN and the flags False (m is always set).

    Under an outer ``vmap`` the cond becomes a select and both branches run:
    correct, but without the saving. Call it on the flat batch.
    """
    N = w6.shape[0]
    K = max(16, N // 16) if size is None else size
    m, near_l, near_n = jax.vmap(_candidates)(lim, D2)
    cand = near_l | near_n

    def one(w: Array, d2: Array, lim1: Limits) -> Taylor:
        return taylor_coeffs(make_F(phi, integrals_of_motion(phi, w, d2)), lim1)

    def mask(t: Taylor, keep: Array) -> Taylor:
        def nan_unless(x: Array) -> Array:
            return jnp.where(keep, x, jnp.nan)

        return Taylor(
            t.use_l & keep,
            m,
            nan_unless(t.c2),
            nan_unless(t.c3),
            nan_unless(t.c4),
            t.use_n & keep,
            nan_unless(t.F2),
            nan_unless(t.F4),
        )

    def full(_: None) -> Taylor:
        return mask(jax.vmap(one)(w6, D2, lim), cand)

    def compact(_: None) -> Taylor:
        idx = jnp.nonzero(cand, size=K, fill_value=N)[0]

        def take(x: Array) -> Array:
            return x.at[idx].get(mode="clip")

        t = jax.vmap(one)(take(w6), take(D2), jax.tree.map(take, lim))
        nans, false = jnp.full(N, jnp.nan, m.dtype), jnp.zeros(N, bool)
        init = Taylor(false, m, nans, nans, nans, false, nans, nans)
        out = Taylor._make(
            a.at[idx].set(x, mode="drop") for a, x in zip(init, t, strict=True)
        )
        return out._replace(m=m)

    out: Taylor = jax.lax.cond(jnp.sum(cand) <= K, compact, full, None)
    return out

"""Staeckel-fudge turning points (per point, then ``vmap``).

The fused turning-point solve, its implicit-function tangents, and the
Taylor coefficients of the Q-form regimes.

Derivatives are first order only. The limits are solved on frozen inputs
and their tangents reattached (`attach_tangents`), so a second derivative
(``jax.hessian``, a jvp of a grad) misses the d2F/dL dtheta and dF' terms
and is silently wrong (MN, a lam_min Hessian: 231 against 310). Any public
entry point built on this core must say so.
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
from ._utils import chandrupatla, clamp, safe_div, tang

# ---------------------------------------------------------------------------
# Turning points. Values only, on frozen inputs; the
# implicit-function tangents are attached separately.

N_ROOT_ITER = 16  # fused Chandrupatla iterations
MINIMUM_RANGE = 1e-10  # C++: smallest nu / lambda interval treated as non-zero
FIRST_STEP_EPS = 1e-3  # clip of the quadratic first step, as a bracket fraction
SNAP_NOISE = 8.0  # snap a double root within this factor of F's roundoff
LADDER = 10.0  # geometric ratio of the outer-end ladders (see find_limits)
N_LADDER_LO, N_LADDER_HI = 2, 7  # ladder points below and above the vertex
LADDER_NEAR = 0.5  # one point each side, past the near root (see find_limits)
LAM_LADDER_D2 = (1, 2, 3, 6, 9)  # lam_lower + (v or lam_pos - lam_lower) 10^-k
NU_LADDER = (1, 2, 3, 6, 9, 12, 15)  # nu ladder: nu_upper + (|nu| - nu_upper) 10^-k
NU_LADDER_SPREAD = (0.1, 0.2, 0.4, 0.6, 0.8)  # ... and b (t / b)^q below it,
NU_LADDER_LO = (1.1, 1e2, jnp.inf, jnp.inf, jnp.inf)  # at most b times these,
NU_LADDER_HI = (0.0, 0.0, 10**-2.5, 10**-1.3, 10**-0.9)  # and at least t times these
LAM_LADDER_FILL = (1 / 3, 2 / 3, 3 / 2)  # lam_lower + (v or lam_pos - lam_lower) 10^-k
LADDER_FAR = (8.0, 24.0, 72.0)  # log(tau / v) of the far ladder points above v
D2_SIDE = 1e-15  # L_z = 0: box vs tube is decided at D2 (1 -+ D2_SIDE)
NU_TP_STEP = 1e-8  # sigma dot = 0: F is read this far from |nu| to nu_upper


class Limits(NamedTuple):
    """Turning points and how each was found (see `find_limits`)."""

    lam_min: Array
    lam_max: Array
    nu_max: Array
    shell: Array  # lam_max - lam_min <= (lam - D2) MINIMUM_RANGE: both set to lam
    planar: Array  # nu_max <= MINIMUM_RANGE D2 (lam - D2) / lam: set to 0
    nu_pinned: Array  # L_z = 0 box/tube decision set nu_max to D2
    # lam_min set to D2: an L_z = 0 decision, an edge root at or below D2, or
    # an edge lam_max lane at lam = D2 (any L_z)
    lmin_pinned: Array
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


def _quad_roots(f: Array, df: Array, d2f: Array) -> tuple[Array, Array, Array]:
    """Offsets (below, above) of the roots of f + f' x + f'' x^2 / 2, and q.

    Cancellation-free: q / (f''/2) and f / q, q = -(f' + sgn(f') sqrt(disc))/2,
    disc = f'^2 - 2 f f''. At a turning point (f ~ 0) the far root is the
    first; 2f / (-f' -+ sqrt(disc)) would be 0/0 there. q is NaN where
    disc < 0 (or NaN); the roots are NaN or inf there or where q = 0.
    """
    disc = df * df - 2 * f * d2f
    sq = jnp.sqrt(jnp.where(disc >= 0, disc, jnp.nan))
    q = -0.5 * (df + jnp.where(df >= 0, sq, -sq))
    r1, r2 = q / (0.5 * d2f), f / q
    return jnp.minimum(r1, r2), jnp.maximum(r1, r2), q


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

    The noise model bounds F's roundoff only for a Phi without internal
    cancellation: |Phi| stands in for the size of Phi's own terms, so where
    they cancel (a central hole) Phi's roundoff is larger than it says.
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
    lo, hi, _ = _quad_roots(f, df, d2f)
    x = lam + (lo if sign < 0 else hi)
    if lam_pos is not None:
        ell = jnp.log(x / lam_pos)
        x = ell / (1 + ell)
    accept = (d2f < 0) & (a < x) & (x < b)
    c_lo, c_hi = a + FIRST_STEP_EPS * (b - a), b - FIRST_STEP_EPS * (b - a)
    return jnp.where(accept, clamp(x, c_lo, c_hi), 0.5 * (a + b))


def default_phi_inf(phi: Callable[[Array, Array], Array], iomz: IOM) -> Array:
    """Phi at the lambda_max lane's guard point, log tau = 230, on nu = 0."""
    R = jnp.sqrt(jnp.exp(LOG_TAU_MAX) - iomz.D2)
    return phi(R, jnp.zeros_like(R))


def _ladder_side(
    v: Array, fv: Array, t: Array, f: Array, *, desc: bool
) -> tuple[Array, Array, Array, Array, Array]:
    """Bracket on one side of the vertex v from the ladder ``t``.

    On [v, t sorted by key], the key t (ascending) or -t (``desc``) with NaN
    keys last, the leading run of F > 0 points ends at the inner end (F > 0,
    nearest the root); the next point, if F < 0 there, is the outer end.
    Returns (inner, f(inner), outer, f(outer), closed). The first point
    in key order without F > 0 is the outer end; the inner end is the
    largest key below it with F > 0, else v; with no such point the run ends
    at the first NaN key (open) or the last point (not closed).
    """
    key = -t if desc else t
    isn = jnp.isnan(key)
    inf = jnp.inf
    kb = jnp.where(~isn & ~(f > 0), key, inf)
    i_b = jnp.argmin(kb)
    has_bad = kb[i_b] < inf
    ki = jnp.where(~isn & (f > 0) & (key < kb[i_b]), key, -inf)
    i_i = jnp.argmax(ki)
    has_in = ki[i_i] > -inf
    any_nan = jnp.any(isn)
    i_last = jnp.argmax(jnp.where(isn, -inf, key))
    vpos = fv > 0
    t_in = jnp.where(vpos & has_in, t[i_i], v)
    f_in = jnp.where(vpos & has_in, f[i_i], fv)
    nan = jnp.nan
    t_out = jnp.where(
        ~vpos, v, jnp.where(has_bad, t[i_b], jnp.where(any_nan, nan, t[i_last]))
    )
    f_out = jnp.where(
        ~vpos, fv, jnp.where(has_bad, f[i_b], jnp.where(any_nan, nan, f[i_last]))
    )
    closed = jnp.where(vpos, has_bad & (f[i_b] < 0), fv < 0)
    return t_in, f_in, t_out, f_out, closed


def find_limits(Fz: Fset, iom: IOM, phi_inf: Array) -> Limits:
    """lam_min, lam_max, nu_max: a branchless port of the C++ ``find_limits``.

    ``Fz`` and ``iom`` must be frozen (built from stop-gradient inputs and a
    stop-gradient potential); the outputs carry no gradient. The three roots
    are found by one fused Chandrupatla loop:

    - nu_max: log tau in [log |nu|, log nu_upper] (|nu| floored at
      MINIMUM_RANGE max(nu_tiny, D2 MINIMUM_RANGE^2)), tightened along a
      ladder towards nu_upper; |nu| if F(|nu|) <= 0 or no bracket.
    - lam_min: log tau in [log a_lam, log b_lam].
    - lam_max: s in [0, b_s], tau = lam_s exp(s/(1-s)), on F/tau^2, and
      E - phi_inf past the guard.

    By default b_lam = lam_s is the nudged lambda, a_lam = lam_lower and
    b_s = 1. Where F's quadratic model at lambda has a maximum v with
    F(v) > 0, the brackets are split at v and tightened along ladders of
    points either side; unsplit, the lam_max bracket is closed from the
    first point above lam_pos without F > 0 (see the comments in the body).
    lam_s is returned as ``lam_pos``.

    At the lambda edge (F(lam) <= 0, or the Newton step from lam shorter
    than (lam - D2) MINIMUM_RANGE) the limit on the side F' points away from
    is not solved: it is lam plus the near root of F's quadratic model at
    lam, with F(lam) taken from the momenta (Newton where the model has no
    real root). The Newton step in the edge test takes the exact F(lam)
    from the momenta. gala sets it to lam itself, up to lam MINIMUM_RANGE
    off: percents of the extent in a deep core, and many times lam_min - D2
    near the axis. gala's threshold is lam MINIMUM_RANGE; with a fixed Delta in a
    deep core (lam - D2 ~ 1e-10 lam) that is the whole extent, and the
    model's root is poor on a cusp (NFW: up to 4.8e-3 of the extent).
    Where F' = 0 both limits are assigned (a double root at lam), except at
    lam = D2 with F'' > 0, where only lam_min is. At lam = D2 an F' within
    its roundoff counts as 0 (the z axis inside the foci with a small v_R).
    At lam = D2, at any L_z, a resolved F < 0 between D2 and lam_pos (a
    cusp centre) makes lam_max F's first root above lam: D2 itself, or a
    root below the window where F is resolved positive at D2 (1 +
    D2_SIDE).

    Snapped (a double root, lam_min = lam_max = lam): on any lane where the
    model's root pair lies within F's own roundoff (`_noise_snap`). This
    replaces gala's ``F(lam_pos) < 0`` clause, which reads the sign of a
    roundoff-level F and so depends on how the batch is compiled. Also
    where no sampled point starts the solved lane: no split and F(lam_pos)
    <= 0, so F > 0 near lam is within F's roundoff. At lam = D2, and where
    lam_pos is at or below lam_lower (L_z = 0 with lam within D2_SIDE of
    D2, where F(lam_pos) is NaN), the lane starts from D2 (1 + D2_SIDE)
    instead, and is snapped only if F <= 0 there too. One case is still
    open: on the axis inside the foci at L_z = 0, F > 0 can be resolved
    further above D2 while F at D2 (1 + D2_SIDE) and at lam_pos is
    roundoff. Those lanes are snapped to D2 as well (audit 12 F1, deferred
    to a follow-up). F(lam_pos) is the quadratic model's value from the
    exact F(lam) where the sampled value agrees with it to F's roundoff.

    A lambda lane without a valid bracket returns NaN (the point is then
    masked); unlike the C++, the MINIMUM_RANGE collapse keeps that NaN. An
    invalid point (``iom.valid`` False) or a non-finite E + I3 + L_z (C++:
    ``isfinite(E + I3 + Lz)``) returns NaN for all three limits, flags False.
    """
    D2, lam, absnu, E = iom.D2, iom.lam, iom.absnu, iom.E
    # L_z^2, not L_z, decides L_z = 0: an L_z that squares to 0 (|L_z| below
    # ~1e-154) is 0 in F too, and F(D2) = 0 then needs the L_z = 0 logic.
    Lz2 = iom.Lz**2
    FD2 = -Lz2 * D2 / 2  # F(D2), on either branch
    Dm, Dp = D2 * (1 - D2_SIDE), D2 * (1 + D2_SIDE)
    # L_z = 0: F(D2) = 0, so box vs tube is decided just either side of D2,
    # and from the ladders (nu_pinned and lmin_pinned, below).
    Lz0 = Lz2 == 0
    nu_upper = jnp.where(Lz0, Dm, D2)
    # the nu ladder starts at |nu|, floored (see the nu lane below); nu_max
    # <= nu_tiny is planar (at the end)
    nu_tiny = MINIMUM_RANGE * D2 * (lam - D2) / lam
    nu_floor = MINIMUM_RANGE * jnp.fmax(nu_tiny, D2 * MINIMUM_RANGE**2)
    nu_lo = jnp.where(absnu > nu_floor, absnu, nu_floor)
    # With d sigma/dt = 0 the point is at a nu turning point and F(|nu|) is
    # 0 up to roundoff, so F is read at t_tp instead, NU_TP_STEP of the way
    # from |nu| to nu_upper (`nu_low`, below).
    sig0 = iom.sigdot == 0
    t_tp = nu_lo + (nu_upper - nu_lo) * NU_TP_STEP

    # F at |nu| (t_tp where sigma dot = 0), D2 (1 -+ D2_SIDE) and lam: one
    # phi call.
    t4 = jnp.stack([jnp.where(sig0, t_tp, absnu), Dm, Dp, lam])
    Phi4, mult4, I4 = _lane_terms(Fz, D2, t4)
    fnu, fm, fp, fl = _F_val(iom, t4, Phi4, mult4, I4)
    dfl = Fz.dF_lam(lam)
    fnu_upper = jnp.where(Lz0, fm, FD2)
    lam_lower, flam_lower = jnp.where(Lz0, Dp, D2), jnp.where(Lz0, fp, FD2)

    # The lambda edge: F(lam) <= 0 or a tiny Newton step. Assign the limit
    # on the side F' points away from, and nudge lam_pos into F > 0.
    # F' = 0 (a flat F, e.g. a maximum below 0 by roundoff): no Newton step,
    # so lam_pos stays finite (it is exported): `safe_div`, not 0/0.
    # "Tiny" is relative to lam - D2, not lam (gala): F's roundoff and a core
    # orbit's extent both scale with lam - D2.
    S = jnp.abs(E * lam) + jnp.abs(I4[3]) + jnp.abs(Phi4[3] * mult4[3])
    eps = jnp.finfo(lam.dtype).eps  # type: ignore[no-untyped-call]
    noise = SNAP_NOISE * eps * (S * jnp.abs(lam - D2) + Lz2 * lam / 2)
    # At lam = D2 (the axis inside the foci) F'(D2) is O(v_R^2), and with a
    # small v_R it is a sum of terms that cancel: `dF_lam`'s limit E D2 -
    # L_z^2/2 - (I_3 - f_nu) - (D2 - |nu|) Phi(0, sigma). I_3 and f_nu each
    # carry lam Phi(rho, 0), Phi at the centre here: it cancels in I_3 - f_nu
    # but sets the roundoff, and it can dwarf the other terms. Within
    # SNAP_NOISE x eps of the size of these terms F' is set to 0, so the
    # edge sides below and the nudge do not follow its sign, which changes
    # with jit, the batch and the AD mode.
    at_D2 = ~(lam > D2)
    S_D2 = (
        jnp.abs(E * D2)
        + jnp.abs(iom.I3)
        + jnp.abs(iom.fnu)
        + jnp.abs(iom.flam)
        + jnp.abs(Phi4[2] * mult4[2])
        + Lz2 / 2
    )
    dfl = jnp.where(at_D2 & (jnp.abs(dfl) <= SNAP_NOISE * eps * S_D2), 0.0, dfl)
    flat = dfl == 0
    dx = safe_div(-fl, dfl)
    tiny = (lam - D2) * MINIMUM_RANGE
    # The tiny-step test takes F(lam) from the momenta, (lam - |nu|)^2
    # lamdot^2 / (8 lam): exact and >= 0, where `fl` is a difference of O(1)
    # terms whose roundoff changes with the compiled batch shape. Ring +
    # disk, R = 10, v_R = 8e-10: F(lam) = 3.2e-15, and `fl` was 0 in eager
    # and 6.6e-14 under jit, so only eager took the edge.
    f_mom = (lam - absnu) ** 2 * iom.lamdot**2 / (8 * lam)
    edge = (fl <= 0) | (~flat & (f_mom < tiny * jnp.abs(dfl)))
    # The nudge is min(half the vertex offset, lam MINIMUM_RANGE), or the
    # Newton step dx if larger, F(lam) <= 0 and F' > 0 (for F' < 0, dx <= 0
    # and fmax(dx, off) is off). F''(lam) is 0/0 = NaN at lam = D2 (R = 0
    # inside the foci): the nudge is lam MINIMUM_RANGE there, and the model's
    # F'' (below) is taken at the nudged point instead. The nudge is at
    # least noise / |F'|, short of half the vertex offset: near a turning
    # point of a near-circular orbit F' is small, and a shorter nudge leaves
    # F(lam_pos) within F's roundoff.
    sgn = jnp.where(dfl >= 0, 1.0, -1.0)
    d2f = Fz.d2F_lam(
        jnp.where(at_D2, lam + jnp.fmax(dx, lam * MINIMUM_RANGE) * sgn, lam)
    )
    # The limit on the side F' points away from is assigned (below). Where
    # F' = 0 both sides are, a double root at lam, except at lam = D2 with
    # F'' > 0 above it (taken there, sgn = 1, `rise_D2`): F > 0 on (D2,
    # lam_max) and only lam_min (D2) is assigned, unless F has a resolved
    # negative below lam_pos (`win_D2`, below). With F'' < 0, F < 0 above
    # D2 and both limits are D2. `rise_D2` also sets the ladder (below).
    rise_D2 = flat & at_D2 & (d2f > 0)
    lmin_edge = edge & (dfl >= 0)
    off = jnp.where(
        at_D2,
        lam * MINIMUM_RANGE,
        jnp.fmin(
            0.5 * jnp.abs(dfl / d2f),
            jnp.fmax(lam * MINIMUM_RANGE, safe_div(noise, jnp.abs(dfl))),
        ),
    )
    # Where F(lam) > 0 the Newton step points at the limit being assigned,
    # away from F > 0: the nudge is then off alone (in a deep core the step
    # can cross the whole extent, past lam_min or below D2). Off the edge
    # with F(lam) <= noise lam_pos is nudged too (both lanes are solved):
    # F(lam) there is F's roundoff, and unnudged the lane would have no
    # F > 0 start and be snapped.
    lam_pos = jnp.where(
        edge | (fl <= noise),
        lam + jnp.where(fl > 0, off, jnp.fmax(dx, off)) * sgn,
        lam,
    )

    # A bracket whose end |F| dwarfs F near the root makes interpolation
    # creep: near a lambda turning point (lam_pos is almost the other root),
    # near the axis inside the foci, at a cusp, and wherever a root is
    # decades from lambda. So split the brackets at the model's vertex v
    # (F's maximum) and close each half from a ladder of points:
    # - above v: v + (o - v) LADDER^k, o = 2r - v (the model's root r
    #   reflected about v; k = 0 is o itself), and the far points lam_pos
    #   exp(LADDER_FAR) for roots decades out;
    # - below v: v - (v - o) LADDER^k, and lam_lower + (v - lam_lower)
    #   LADDER^-k (k in LAM_LADDER_D2) towards a root just above D2 (tiny
    #   L_z, near the axis). Where F'' >= 0 at lam or v is at or below
    #   lam_lower (no split) these run from lam_pos instead of v.
    # - one point on each side past the model's near root r, by half of |r -
    #   lam|: below v the reflected points are often below lam_lower and the
    #   next is a decade away, and above v o can land past a thin F < 0
    #   window in a second F > 0 interval (an explicit Delta about as large
    #   as the orbit, or a non-monotone Phi).
    # On each side the inner end is the last point of the leading run with
    # F > 0, the outer end the next point if F < 0 there (else lam_lower, or
    # open). Unsplit, the lam_max lane is closed from the points above
    # lam_pos (above v and far), and the lam_min lane from lam_pos and the
    # points towards D2. On the axis outside the foci at L_z = 0 (|nu| =
    # D2) a small v_R puts lam_min ~1e-13 to 1e-5 (lam - D2) above D2,
    # where F is tiny: hence the ladder's run to 10^-9.
    # The nu bracket has the same creep where nu_max is near nu_upper (F
    # there is tiny next to F(|nu|)): it is closed from nu_upper + (|nu| -
    # nu_upper) LADDER^-k (k in NU_LADDER) and nu_upper itself. Near D2, F ~
    # c (tau - nu_max)(tau - D2): the ladder runs to 10^-15 so the root's
    # segment spans at most 10^3 in D2 - tau. It is also closed from five
    # points below those, in [b, t]: t is the ladder's first point and b =
    # max(|nu|, nu_tiny) (a root below nu_tiny is planar, and at z = 0 |nu|
    # is floored at ~1e-20 D2). They are b (t / b)^q (q in
    # NU_LADDER_SPREAD), at most b NU_LADDER_LO and at least t NU_LADDER_HI.
    # Where t / b < 100 that is 1.1 b and four points a fifth of the span
    # apart in log tau. The bracket [|nu|, nu_upper] spans up to ~70 decades
    # at a small nu_max (nu_tiny is ~1e-30 D2 in a core), and the solver
    # creeps on a root just above |nu| (near the nu turning point): hence
    # 1.1 b and 100 b (with 10 b the first point, nu_max was 5e-4 off on a log
    # potential at 1.6% above |nu|). The F < 0 windows that were missed sat
    # in the top three decades below t: hence t 10^-2.5, t 10^-1.3 and t
    # 10^-0.9, placed to catch the windows of the tests and audits. The
    # earlier multiples of |nu| left gaps of 1.2 to 5 decades, and a window
    # in one was skipped (three masses on the z axis at z = 0, D2 = 647.7:
    # nu_max 5.42 against 8.0e-5; a disk plus a flattened core, D2 = 1.97:
    # 1.97 against 2.6e-2). A window narrower than the gap it falls in is
    # still missed (0.35 decade just above 1.1 |nu|, the three masses at D2
    # = 133).
    # F at lam_pos, v and every ladder point: one phi call.
    xl, xh, _ = _quad_roots(fl, dfl, d2f)
    v = lam - dfl / d2f
    # Where v is at or below lam_lower (no split) the points above run from
    # lam instead: lam + 2 x_hi LADDER^k. Run from v, far below lam, the
    # first point could land past the orbit's F < 0 side, in a second F > 0
    # interval (the lam_max ladder, below).
    v_hi = jnp.where(v > lam_lower, v, lam)
    dl, dh = v - (2 * (lam + xl) - v), 2 * (lam + xh - v_hi)
    g_lo = LADDER ** jnp.arange(N_LADDER_LO, dtype=lam.dtype)
    g_hi = LADDER ** jnp.arange(N_LADDER_HI, dtype=lam.dtype)
    # `rise_D2`: F ~ (tau - D2)^2 near D2 and the model has no root to scale
    # the ladder (x_hi is NaN at L_z = 0). lam_max - D2 is anywhere from
    # ~1e-10 D2 (v_z just past what reaches a focus) to decades out, so the
    # points above run lam + lam MINIMUM_RANGE LADDER^2k, up to 100 lam.
    # The same ladder is used where lam - D2 <= lam MINIMUM_RANGE and the
    # model's step dh is below lam MINIMUM_RANGE or NaN. On the axis inside
    # the foci with a small v_R or L_z, F'(D2) > 0 is resolved and the
    # model's root above lam is within an ulp of lam, so dh was 0 and only
    # lam_pos e^8 closed the lane. With
    # lam_max - D2 ~ 1e-3 D2 near the bottom of that bracket, 16 iterations
    # left lam_max 2.2e-6 of the extent off (Miyamoto-Nagai, D2 = 1, R =
    # 1e-8), 1.6e-2 off (Plummer, D2 = 1e-4, R = 1e-10) and 4.1e-2 off
    # (isochrone, the same point, under jit). Away from D2 the ladder is
    # not used. Near a circular orbit F's minimum near lam can be within
    # its roundoff, and a roundoff F < 0 at lam (1 + 1e-10) closed the lane
    # (ring + disk, R = 4.16: lam_max 21.30 against 103.48). With F'' > 0
    # and real model roots only, its 100-fold steps skipped a thin F < 0
    # window just above lam (ring + cusp, D2 = 213.6, lam - D2 = 4.5e-7:
    # lam_max 278.87 against 216.46). Where the model gives no step above
    # lam (dh <= 0 or NaN: F'' > 0 with both roots at or below lam, or no
    # real root) the points above run lam + h LADDER^k instead, h = max(lam
    # - D2, 1e-8 lam), so they reach at least 1e-2 lam: only lam_pos e^8
    # closed those lanes, and 16 iterations left lam_max far off (two tori,
    # D2 = 3000: 3000.13 against 3003.07). The rise ladder there skipped
    # the ring + cusp window above.
    near_D2 = ~(lam - D2 > lam * MINIMUM_RANGE)
    rise = rise_D2 | (near_D2 & ~(dh >= lam * MINIMUM_RANGE))
    dh = jnp.where(
        rise,
        lam * MINIMUM_RANGE,
        jnp.where(dh > 0, dh, jnp.fmax(lam - D2, lam * 1e-8)),
    )
    g_hi = jnp.where(rise, g_hi * g_hi, g_hi)
    g_d2 = LADDER ** -jnp.asarray(LAM_LADDER_D2, dtype=lam.dtype)
    # The near-root points. Where F'' < 0 the near roots are r_lo and r_hi
    # and the step is at least LADDER_NEAR w/4 (w = r_hi - r_lo): with lam
    # at a turning point one root is lam itself and F just past it is
    # roundoff. Otherwise r is the model's root nearest lam on that side.
    conc = d2f < 0
    x_n = jnp.where(conc | ~(xh < 0), xl, xh)
    x_p = jnp.where(conc | ~(xl > 0), xh, xl)
    w_q = jnp.where(conc, 0.25 * (xh - xl), 0.0)
    t_n = lam + x_n - LADDER_NEAR * jnp.fmax(-x_n, w_q)
    t_p = lam + x_p + LADDER_NEAR * jnp.fmax(x_p, w_q)
    # The near-root point below and the two points below v are empty (NaN
    # or at or below lam_lower) where the model has no root above lam_lower
    # on that side. Those slots then hold lam_lower + (top - lam_lower)
    # 10^-k (k in LAM_LADDER_FILL), top = v or lam_pos as for the ladder to
    # lam_lower. Unsplit, the next point below lam_pos was a decade down: a
    # ring potential (L_z = 0, D2 = 0.127) gave lam_min = D2 against 0.676
    # (F(v) < 0 was sampled there but is not a lam_min point), and four more
    # ring lanes a root near D2 or D2 itself.
    # Split, a window between the first two points below v was skipped (D2
    # = 0.077: D2 against 0.278).
    top_lo = jnp.where(conc & (v > lam_lower), v, lam_pos)
    g_fill = LADDER ** -jnp.asarray(LAM_LADDER_FILL, dtype=lam.dtype)
    t_fill = lam_lower + (top_lo - lam_lower) * g_fill
    t_n3 = jnp.concat(
        [
            jnp.where(conc | (x_n < 0), t_n, jnp.nan)[None],
            jnp.where(dl > 0, v - dl * g_lo, jnp.nan),
        ]
    )
    t_n3 = jnp.where(t_n3 > lam_lower, t_n3, t_fill)
    t_far = lam_pos * jnp.exp(jnp.asarray(LADDER_FAR, dtype=lam.dtype))
    tk = jnp.concat(
        [
            lam_pos[None],
            v[None],
            # i_lo: below v (or lam_pos)
            t_n3,
            lam_lower + (top_lo - lam_lower) * g_d2,
            # i_hi: above v (or lam)
            jnp.where(conc | (x_p > 0), t_p, jnp.nan)[None],
            jnp.where(dh > 0, v_hi + dh * g_hi, jnp.nan),
            jnp.where(jnp.log(t_far) < LOG_TAU_MAX, t_far, jnp.nan),
        ]
    )
    # the points as F sees them: NaN at or below lam_lower (its F is NaN, and
    # read as F <= 0 it would close a bracket on a NaN end)
    tk = jnp.where(tk > lam_lower, tk, jnp.nan)
    n_k = len(tk)
    i_lo = slice(2, 3 + N_LADDER_LO + len(LAM_LADDER_D2))
    i_hi = slice(i_lo.stop, n_k)
    t_lo, t_hi = tk[i_lo], tk[i_hi]
    g_nu = LADDER ** -jnp.asarray(NU_LADDER, dtype=lam.dtype)
    t_nu = nu_upper + (nu_lo - nu_upper) * g_nu
    b_nl = jnp.fmax(nu_lo, nu_tiny)
    r_nl, q_nl = t_nu[0] / b_nl, jnp.asarray(NU_LADDER_SPREAD, dtype=lam.dtype)
    t_nl, c_lo = b_nl * r_nl**q_nl, b_nl * jnp.asarray(NU_LADDER_LO, dtype=lam.dtype)
    t_nl = jnp.where(t_nl < c_lo, t_nl, c_lo)
    c_hi = t_nu[0] * jnp.asarray(NU_LADDER_HI, dtype=lam.dtype)
    t_nl = jnp.where(t_nl > c_hi, t_nl, c_hi)
    t_nl = jnp.where(t_nl < t_nu[0], t_nl, jnp.nan)  # only below the ladder
    t_nu = jnp.concat([t_nl, t_nu, nu_upper[None]])
    t_all = jnp.concat([tk, t_nu[:-1]])
    Phik, multk, Ik = _lane_terms(Fz, D2, t_all)
    fk = _F_val(iom, t_all, Phik, multk, Ik)
    fk, f_nu = fk[:n_k], jnp.concat([fk[n_k:], fnu_upper[None]])
    f_lp, f_v, f_lo, f_hi = fk[0], fk[1], fk[i_lo], fk[i_hi]
    # Where the sampled F(lam_pos) is within F's roundoff of the quadratic
    # model from f_mom, the model value is used: its sign does not change
    # with the compile. Ring + disk, R = 4.16, v_R = 4e-9: F has a minimum
    # of 3.3e-15 near lam and no root there, but F(lam_pos) was -8.5e-16 in
    # a 16-lane batch, and the lane was snapped. At lam = D2 the model does
    # not hold (f_mom is not F(D2)).
    x_lp = lam_pos - lam
    m_lp = f_mom + x_lp * (dfl + 0.5 * d2f * x_lp)
    f_lp = jnp.where(~at_D2 & (jnp.abs(f_lp - m_lp) <= noise), m_lp, f_lp)

    # L_z = 0: nu_max is pinned to D2 unless F is a resolved negative
    # (below -SNAP_NOISE x its roundoff eps (S_tau + S) |tau - D2|, S_tau
    # and S the sizes of its terms at tau and at lam) at D2 (1 -
    # D2_SIDE) or at a nu ladder point above |nu|: that is a root below D2.
    # On the z axis F_nu has a double zero at D2 (inside the foci at lam =
    # D2, and outside at rest), F near D2 is roundoff, and its sign would
    # change with the batch shape.
    # ``lmin_pinned`` (below) is the mirror on the lambda side.
    def resolved_neg(f: Array, t: Array, Phi: Array, mult: Array, I: Array) -> Array:
        S_t = jnp.abs(E * t) + jnp.abs(I) + jnp.abs(Phi * mult)
        return f < -SNAP_NOISE * eps * (S_t + S) * jnp.abs(t - D2)

    S_k = jnp.abs(E * t_nu[:-1]) + jnp.abs(Ik[n_k:]) + jnp.abs(Phik[n_k:] * multk[n_k:])
    thr_k = SNAP_NOISE * eps * (S_k + S) * jnp.abs(t_nu[:-1] - D2)
    sgn_k = jnp.stack([f_nu[:-1] < -thr_k, f_nu[:-1] > thr_k])
    # the lowest ladder points above |nu| with F resolved negative and positive
    k_nu = jnp.where((t_nu[:-1] > nu_lo) & sgn_k, t_nu[:-1], jnp.inf)
    m_neg, m_pos = jnp.min(k_nu, 1)
    neg_nu = m_neg < jnp.inf
    neg_m = resolved_neg(fm, Dm, Phi4[1], mult4[1], I4[1])
    # A phase point with d sigma/dt = 0 is at a nu turning point. A resolved
    # F > 0 at t_tp (`resolved_neg` of -F) makes |nu| the lower one
    # (`nu_low`): the lane is pinned or solved as any other, with F(t_tp)
    # standing in for F(|nu|). Otherwise |nu| is the upper turning point
    # (`nu_top`) and nu_max is |nu| (nu_done), not the pin. Just inside a
    # focus at rest (|nu| ~ D2 (1 - 1e-7)) F between |nu| and D2 is not
    # resolved, so that lane is `nu_top`. A turning point within NU_TP_STEP
    # (nu_upper - |nu|) above a lower one is missed (nu_max is then |nu|).
    # F(0) is not read: F(0) <= 0 means the orbit does not reach the plane
    # (nu has a lower turning point above 0), but its upper turning point is
    # still the first root above |nu|, or D2 at L_z = 0 with no resolved
    # negative (plum2 at L_z = 0, D2 = 1: nu_max 1.0, where F(0) <= 0 had
    # returned |nu| = 1.5e-3).
    # Where F(t_tp) is not a resolved positive (t_tp - |nu| ~ 1e-16 for |nu|
    # = 3.5e-12, D2 = 1e-8), the lowest ladder point above |nu| with F
    # resolved decides: F > 0 there is `nu_low` too (`low_k`). A hole
    # potential at rest gave nu_max = |nu| there against D2. A resolved
    # negative F(t_tp) is not trusted: Phi's own terms cancel in a hole, and
    # F(t_tp) = 1e-27 read as one there (D2 = 5.2e-5). The lane is still
    # read from F(t_tp), so at L_z != 0 (not pinned) an F(t_tp) <= 0 keeps
    # |nu|; using the ladder point instead cost ~1.6k flops in XLA's fusion.
    pos_tp = (t_tp > nu_lo) & resolved_neg(-fnu, t_tp, Phi4[0], mult4[0], I4[0])
    low_k = sig0 & ~pos_tp & (m_pos < m_neg)
    nu_low = (sig0 & pos_tp) | low_k
    nu_top = sig0 & ~nu_low
    nu_pinned = Lz0 & ~(neg_m | neg_nu | nu_top)

    # Snapped: a root pair within F's roundoff (`_noise_snap`; it subsumes
    # gala's near-shell test at the edge, where the model has no real root).
    # F(lam) for the snap is f_mom (above).
    # No F > 0 sample to start the solved lane from (F(lam_pos) <= 0 or NaN,
    # and no split, below): at lam = D2, or with lam_pos at or below
    # lam_lower (L_z = 0, lam - D2 <= D2 D2_SIDE: F(lam_pos) is NaN), it
    # starts from D2 (1 + D2_SIDE) where F > 0 there (`at_D2_low`, below);
    # otherwise F > 0 near lam is within F's roundoff (a core with E -
    # Phi(0) near eps |Phi(0)|, or an extent of a few ulp of D2 at lam =
    # D2), and the lane is snapped. One case is still open: on the axis
    # inside the foci at L_z = 0, F > 0 can be resolved further above D2
    # while F at D2 (1 + D2_SIDE) and at lam_pos is roundoff. Those lanes
    # are snapped to D2 too (audit 12 F1, deferred to a follow-up).
    split_ok = (d2f < 0) & (f_v > 0)  # the split (see below)
    no_pos = ~split_ok & ~(f_lp > 0)
    at_D2_low = no_pos & (at_D2 | ~(lam_pos > lam_lower)) & (fp > 0)
    snapped = _noise_snap(f_mom, dfl, d2f, lam - D2, S, Lz2 * lam / 2)
    snapped = snapped | (no_pos & ~at_D2_low)
    # The lam_min ladder: below v (split), or below lam_pos (unsplit, where
    # F(lam_pos) > 0). Two calls and a select, not one call on selected
    # inputs: the same values, but XLA compiles the latter differently.
    lo_s = _ladder_side(v, f_v, t_lo, f_lo, desc=True)
    t_lo_u = jnp.where(t_lo < lam_pos, t_lo, jnp.nan)
    lo_u = _ladder_side(lam_pos, f_lp, t_lo_u, f_lo, desc=True)
    lo_in, flo_in, lo_out, flo_out, lo_closed = (
        jnp.where(split_ok, s_, u_) for s_, u_ in zip(lo_s, lo_u, strict=True)
    )
    lo_ok = split_ok | (f_lp > 0)
    # L_z = 0: lam_min is pinned to D2 unless F(D2 (1 + D2_SIDE)) or F at a
    # lam_min ladder point below lam_pos is a resolved negative (as for
    # nu_pinned, above). On the axis outside the foci with v_R = 0 (|nu| =
    # D2) F_lam has a double zero at D2 and that F is roundoff. Near the
    # plane where Phi on the axis is deep, F'(D2) > 0: F has a thin F > 0
    # interval just above D2, then a resolved F < 0, then the orbit's. The
    # ladder then closes the bracket below the root nearest lam_pos, split
    # or not (`lo_neg`).
    # A closed ladder bracket is used split or not, at any L_z: it holds the
    # root nearest lam_pos, where [lam_lower, lam_pos] can hold 3 roots.
    fp_neg = resolved_neg(fp, Dp, Phi4[2], mult4[2], I4[2])
    neg_lo = resolved_neg(f_lo, t_lo, Phik[i_lo], multk[i_lo], Ik[i_lo])
    lo_neg = Lz0 & lo_ok & lo_closed & jnp.any(neg_lo & (t_lo < lam_pos))
    near_l = lo_ok & lo_closed
    # `win_D2` at a cusp centre on the axis: F ~ -c (tau - D2)^(3/2) + b (tau
    # - D2)^2 is negative on (D2, D2 + x*) and F'' > 0 past x*, so with x*
    # below lam_pos - D2 = D2 MINIMUM_RANGE F'' there reads as a rise. The
    # rule at lam = D2: lam_max is F's first root above the true lam. lam -
    # D2 = rho^2 rounds to 0 here, so it is below half an ulp of D2 and
    # below every sampled point. `win`: F is a resolved negative at a
    # lam_min ladder point below lam_pos (D2 + 1e-11 D2 down to 1.1e-15 D2,
    # which stands in for D2 (1 + D2_SIDE) too). If F at D2 (1 + D2_SIDE)
    # is also not a resolved positive (`rise_Dp`), the window starts at lam
    # and lam_max is on the edge, at D2 (`win_D2`). If it is, F > 0 runs
    # from lam to a root below the window, and the lam_max lane starts at
    # D2 (1 + D2_SIDE) as for `at_D2_low` (`from_Dp`). This holds at any
    # L_z: NFW, D2 = 1e-8, R = 1e-13 and L_z = 1e-22 to 1e-19 have the
    # window (lam_max D2, and D2 (1 + 3e-13) at L_z = 1e-19), and the lane
    # was solved past it to 2558 while the test was gated on L_z = 0. F
    # must be a resolved negative (as above) and also below the roundoff
    # of F'(D2) (tau - D2), F'(D2) to SNAP_NOISE eps S_D2 (see dfl): near
    # D2 that term sets F's roundoff, which the S_tau + S of
    # `resolved_neg` miss. Hernquist, D2 = 1e6, z = 999: F > 0 above D2,
    # but under jit F at D2 (1 + 1e-13) is negative by 30 times the
    # `resolved_neg` bound and by 1/200 of this one.
    n_D2 = SNAP_NOISE * eps * S_D2
    win = jnp.any(neg_lo & (t_lo < lam_pos) & (f_lo < -n_D2 * (t_lo - D2)))
    rise_Dp = fp > n_D2 * (Dp - D2)
    win_D2 = at_D2 & win & ~rise_Dp
    lmax_edge = edge & (((dfl <= 0) & ~rise_D2) | win_D2)
    from_Dp = at_D2_low | (at_D2 & win & rise_Dp)
    lmin_pinned = (
        (Lz0 & ~fp_neg & ~lo_neg & ~lmin_edge) | (lmax_edge & (lam == D2))
    ) & ~snapped

    # The edge limit (deviation from gala, which sets it to lam: an error up
    # to lam MINIMUM_RANGE, percents of the extent in a deep core): lam plus
    # the near root of the quadratic model built from f_mom, on the side F'
    # points away from (f_mom >= 0); the Newton step -f_mom/F' where the
    # model has no real root; 0 where F' = 0. Cancellation-free (as in
    # `_quad_roots`): |q| >= |F'|/2 > 0 where F' != 0.
    q_m = _quad_roots(f_mom, dfl, d2f)[2]  # NaN where disc < 0 or F'' is NaN
    x_edge = jnp.where(flat, 0.0, jnp.where(jnp.isnan(q_m), -f_mom / dfl, f_mom / q_m))
    lam_edge = jnp.where(snapped, lam, lam + x_edge)  # the value of an edge limit

    # An edge lam_min at or below D2 is pinned there: at L_z = 0 the root is
    # D2 exactly (F(D2) = 0) and the model's fell up to a few ulp below it.
    lmin_pinned = lmin_pinned | (lmin_edge & ~snapped & (lam_edge <= D2))
    lmin_done = lmin_pinned | lmin_edge | snapped
    lmin_set = jnp.where(lmin_pinned, D2, lam_edge)  # value of a done lam_min lane

    # The split and the ladder brackets (see above); positive forms. F(v) > 0
    # puts v between the roots; F(v) > F(lam_pos) is not asked: with lam at
    # F's vertex near the foci the two differ by far less than F's roundoff.
    # Unsplit, the ladder starts at lam_pos and runs over the points above
    # it: the first sign change above lam_pos. From the far points alone it
    # could skip the orbit's root and close on a second F > 0 interval
    # decades out.
    hi = _ladder_side(
        jnp.where(split_ok, v, lam_pos),
        jnp.where(split_ok, f_v, f_lp),
        jnp.where(split_ok | (t_hi > lam_pos), t_hi, jnp.nan),
        f_hi,
        desc=False,
    )
    # `at_D2_low`: lam = D2 (lam - D2 rounds to 0: R <~ 1e-8 Delta inside the
    # foci) with F(lam_pos) <= 0, where the nudge lam MINIMUM_RANGE overshot
    # lam_max (a core orbit's lam_max - D2 ~ R_max^2 is down to ~1e-13 D2
    # there), or lam_pos at or below lam_lower, where F(lam_pos) is NaN. In
    # both lam_min is D2 to an ulp, so F(D2 (1 + D2_SIDE)) > 0 unless the
    # whole extent is within D2_SIDE of D2 (snapped, above): the lane starts
    # there and is closed from the nearest point above with F < 0 among all
    # the sampled points (lam_pos, v, the points below and above v, the far
    # points). A point where F is exactly 0 is skipped: on the axis inside
    # the foci with lam an ulp above D2, F at lam_pos and at the first rise
    # points (D2 (1 + 1e-10), D2 (1 + 1e-8)) rounds to 0 while F ~ (tau -
    # D2)^2 > 0, and that 0 left the bracket open to infinity (isochrone, D2
    # = 1e-4, R = 1e-10 in eager mode, where lam rounds an ulp above D2:
    # lam_max 7.8e-2 of the extent off). `from_Dp` adds the lanes at lam =
    # D2 with a window above an F > 0 run that starts at lam (`win`, above).
    up = _ladder_side(
        Dp, fp, jnp.where((tk > Dp) & (fk != 0), tk, jnp.nan), fk, desc=False
    )
    split, f_split, out_h, fout_h, near_h = (  # split: the lam_max lane's origin
        jnp.where(from_Dp, x_up, x_hi) for x_up, x_hi in zip(up, hi, strict=True)
    )
    lmin_b = jnp.where(lo_ok, lo_in, lam_pos)
    lmin_fb = jnp.where(lo_ok, flo_in, f_lp)
    lmin_a = jnp.where(near_l, lo_out, lam_lower)
    lmin_fa = jnp.where(near_l, flo_out, flam_lower)
    ell = jnp.log(jnp.where(near_h, out_h, 2 * split) / split)
    lmax_b = jnp.where(near_h, ell / (1 + ell), 1.0)
    lmax_fb = jnp.where(near_h, fout_h / out_h**2, E - phi_inf)

    # accept, clip and midpoint are evaluated in tau, then log-mapped below
    lmin_x0 = _first_step(lam, fl, dfl, d2f, lmin_a, lmin_b, sign=-1.0)

    # Lane variables: log tau, log tau (each scaled, below), s. The lam_min
    # lane runs in log tau: with a small (fallback) Delta its bracket spans
    # many decades, and bisection in tau needs ~log2(lam/lam_min) steps
    # before interpolation can take over. So does the nu lane: its bracket
    # [|nu|, D2] spans decades near the centre, and on a cusp (Phi ~ r) F ~
    # a + b sqrt(tau).
    # F(|nu|) <= 0 (at the nu turning point, to roundoff) is |nu| too: the
    # bracket [|nu|, nu_upper] has no sign change (L_z != 0, or L_z = 0 with
    # F(nu_upper) < 0), unless F(D2 (1 - D2_SIDE)) > 0 by roundoff. With
    # d sigma/dt = 0 the point is its own nu turning point and F(|nu|) is 0
    # up to a roundoff whose sign changes with the compile: on an axis lane
    # at rest inside the foci a positive F(|nu|) sent the lane to nu_upper
    # (nu_max ~ D2 against 2.5). So F(|nu|) is not read there: an upper
    # turning point (`nu_top`) is |nu|, and a lower one has F(t_tp) > 0.
    nu_done = nu_pinned | ~(fnu > 0) | nu_top
    done0 = jnp.stack([nu_done, lmin_done, lmax_edge | snapped])
    # The nu lane's |nu| is floored where it is below (e.g. 0 at z = 0) nu_tiny
    # MINIMUM_RANGE (D2 MINIMUM_RANGE^3 at least, for lam = D2): a root below
    # that is collapsed to 0 anyway, or negligible. The bracket's fa stays
    # F(|nu|), not F at the floor: the two differ by F' times the floor, and
    # a sign change between them is a root below the floor. With d sigma/dt
    # = 0 it is F(t_tp) (`nu_low`). The bracket is the ladder's where it
    # closes (above), else [|nu|, nu_upper].
    nu_in, fnu_in, nu_out, fnu_out, nu_closed = _ladder_side(
        nu_lo, fnu, t_nu, f_nu, desc=False
    )
    near_nu = (fnu > 0) & nu_closed
    # The two log lanes run in log(tau / sc), sc = sqrt(a b) the bracket's
    # geometric middle, so |x| is at most half the bracket's span in e-folds
    # and Chandrupatla's stop |b - a| <= 4 eps |x| is a tau-relative
    # tolerance independent of the units. In log tau itself |x| is ~94 in SI
    # (tau ~ 1e41 m^2) and the stop is ~380 ulp of tau: lam_min - D2 off by
    # 2e-3 of the extent near the foci. Each end is log(end / sc), not log
    # end - log sc, which would round to the ulp of log end.
    ta = jnp.stack([jnp.where(near_nu, nu_in, nu_lo), lmin_a])
    tb = jnp.stack([jnp.where(near_nu, nu_out, nu_upper), lmin_b])
    sc = jnp.sqrt(ta) * jnp.sqrt(tb)
    xa, xb = jnp.log(ta / sc), jnp.log(tb / sc)
    x0 = jnp.stack(
        [
            xa[0],
            jnp.log(jnp.where(lmin_done, lmin_set, lmin_x0) / sc[1]),
            _first_step(lam, fl, dfl, d2f, 0.0, lmax_b, sign=1.0, lam_pos=split),
        ]
    )
    a = jnp.stack([xa[0], xa[1], jnp.zeros_like(lam)])
    b = jnp.stack([xb[0], xb[1], lmax_b])
    fa = jnp.stack([jnp.where(near_nu, fnu_in, fnu), lmin_fa, f_split / split**2])
    fb = jnp.stack([jnp.where(near_nu, fnu_out, fnu_upper), lmin_fb, lmax_fb])

    def lanes(x: Array) -> Array:
        sem = _semi_tau(x[2], split)
        tau = jnp.stack([sc[0] * jnp.exp(x[0]), sc[1] * jnp.exp(x[1]), sem[0]])
        Phi, mult, I = _lane_terms(Fz, D2, tau)
        F = _F_val(iom, tau[:2], Phi[:2], mult[:2], I[:2])
        Fs = _F_semi_val(iom, sem, Phi[2], mult[2], I[2], phi_inf)
        return jnp.concat([F, Fs[None]])

    r = chandrupatla(lanes, a, b, fa, fb, x0, done0, N_ROOT_ITER)

    # sc exp(x) can round an endpoint root an ulp outside the bracket
    # (lam_min: below D2 for tiny L_z): clamp back into it. nu_max: NaN (no
    # bracket) is |nu| (C++ fallback).
    nu_tau = clamp(sc[0] * jnp.exp(r[0]), absnu, nu_upper)
    # A pinned lane is D2, also where F(D2 (1 - D2_SIDE)) < 0 is
    # unresolved.
    nu_max = jnp.where(
        nu_pinned,
        D2,
        jnp.where(~nu_done & jnp.isfinite(nu_tau), nu_tau, absnu),
    )
    lmin_tau = clamp(sc[1] * jnp.exp(r[1]), lmin_a, lmin_b)
    lam_min = jnp.where(lmin_done, lmin_set, lmin_tau)
    ts, beyond, ok = _semi_tau(r[2], split)
    lam_max = jnp.where(done0[2], lam_edge, jnp.where(ok & ~beyond, ts, jnp.nan))

    # Ignore extremely small intervals: a shell where lam_max - lam_min is
    # at most (lam - D2) MINIMUM_RANGE, planar where nu_max is at most
    # nu_tiny ((z/R)^2 at most MINIMUM_RANGE). Deviation from gala, whose D2
    # MINIMUM_RANGE (both) is the whole extent of a fixed-Delta core orbit
    # (r <~ 1e-5 Delta); far from the foci the planar test is gala's.
    # Positive form for shell: NaN stays NaN.
    ok = iom.valid & jnp.isfinite(E + iom.I3 + iom.Lz)
    planar = ok & ~(nu_max > nu_tiny)
    shell = ok & (lam_max - lam_min <= (lam - D2) * MINIMUM_RANGE)
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
    with a tiny L_z (nu_max or lam_min rounds to D2): the zeroed theta
    tangent is then O(L_z) (measured ~3e-9 relative). Such a limit still
    moves with D2; `attach_tangents` gives it D2's own tangent.
    """
    ok = jnp.isfinite(L0) & (L0 != D2)
    return jnp.where(ok, tang(Fn(jnp.where(ok, L0, safe))), 0.0)


def _over(num: Array, den: Array) -> Array:
    """Num / den, and 0 (NaN-free) where den is 0 or non-finite."""
    ok = jnp.isfinite(den) & (den != 0)
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)


def _double_tangent(F: Fset, Fz: Fset, lim: Limits) -> Array:
    """-tang(F'(L0)) / F''(L0) at L0 = lam_max on shell lanes, 0 elsewhere."""
    sg = jax.lax.stop_gradient
    lmax = sg(lim.lam_max)
    d = -_over(_tang_at(F.dF_lam, lmax, F.D2, 2 * F.D2), sg(Fz.d2F_lam(lmax)))
    return jnp.where(lim.shell, d, 0.0)


def attach_tangents(lim: Limits, F: Fset, Fz: Fset, *, shell: bool = True) -> Limits:
    """Reattach implicit-function tangents to `find_limits`'s values.

    ``F`` is the live Fset (its tangents are the ones attached), ``Fz`` the
    frozen one (the denominators). Per limit, values unchanged:

    - simple root (solver roots, the edge's ``lam + x_edge`` and the
      ``|nu|`` fallback): ``-tang(F(L0)) / F'(L0)``;
    - double root (shell, incl. snapped): ``-tang(F'(L0)) / F''(L0)``;
    - at D2 (pinned, or rounded onto D2 by a tiny L_z, shell or not): D2's
      own (``tang(D2)``); a pinned nu_max = 0 gets none.

    Every denominator is guarded (0 on snapped lanes, NaN at D2). First order
    only: second derivatives miss the d2F/dL dtheta and dF' terms.

    With ``shell=False`` the shell lanes get no tangent here: the double-root
    tangent needs F' and F'' through a second-order nest on every lane, and
    shell lanes are a subset of the Taylor candidates, so
    `taylor_coeffs_batch` (``shell=True``) adds it on the compacted lanes,
    which is cheaper. Pass it these limits only: limits from the default
    ``shell=True`` would get the shell tangent twice.

    The limits are final only after `taylor_coeffs_batch`: it also
    Newton-polishes the near-circular lambda roots (`_polish`), which are
    roundoff-limited here, and returns the final limits as
    ``Taylor.lam_min`` and ``Taylor.lam_max``.
    """
    D2 = F.D2
    sg = jax.lax.stop_gradient
    lmin, lmax, nu = sg(lim.lam_min), sg(lim.lam_max), sg(lim.nu_max)

    lam2 = jnp.stack([lmin, lmax])
    simple = -_over(_tang_at(F.F_lam, lam2, D2, 2 * D2), sg(Fz.dF_lam(lam2)))
    double = _double_tangent(F, Fz, lim) if shell else 0.0
    t_lam = jnp.where(lim.shell, double, simple)
    t_D2 = tang(D2)  # a traced Delta: limits at D2 move with it
    t_lam = jnp.where(lam2 == sg(D2), t_D2, t_lam)  # incl. pinned lam_min

    t_nu = -_over(_tang_at(F.F_nu, nu, D2, 0.5 * D2), sg(Fz.dF_nu(nu)))
    t_nu = jnp.where(lim.planar | lim.nu_pinned, 0.0, t_nu)
    t_nu = jnp.where(~lim.planar & (nu == sg(D2)), t_D2, t_nu)
    return lim._replace(
        lam_min=lmin + t_lam[0], lam_max=lmax + t_lam[1], nu_max=nu + t_nu
    )


# ---------------------------------------------------------------------------
# Taylor coefficients of the near-shell and near-planar Q.

NEAR_SHELL = 1e-2  # w < NEAR_SHELL hi: quartic-Taylor Q on the lambda branch
NEAR_PLANAR = 1e-4  # nu_max < NEAR_PLANAR D2: candidate for the even-Taylor nu Q
NEAR_PLANAR_SERIES = 3e-5  # ... used iff |F4| nu_max < NEAR_PLANAR_SERIES |F2|
POLISH = 1e-3  # w < POLISH lam: Newton-polish the lambda roots (`_polish`)
N_POLISH = 3  # Newton steps on the quartic
POLISH_SLOPE = 1e-5  # ... kept where F' there is the quartic's to this (a kink)
# A series must match F itself to this (+ F's roundoff) to be used. The
# quartic's own truncation reaches ~4e-7 of F at w/4 by the w = 1e-2 hi switch
# (MW2022), hence the looser near-shell bound; a resolvable kink is off by >~1e-3.
SERIES_CHECK_N, SERIES_CHECK_L = 1e-8, 1e-6


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
    # The final limits: the input's, Newton-polished where near-circular
    # (with ``iom``, as in the batch) and, from `taylor_coeffs_batch(...,
    # shell=True)`, with the shell lanes' double-root tangent.
    lam_min: Array
    lam_max: Array


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


def _candidates(lim: Limits, D2: Array) -> tuple[Array, Array, Array, Array]:
    """(m, near-shell, near-planar candidate, m inside the lambda branch).

    From the limits alone. Both are candidates: use_l and use_n also need
    the series check on the coefficients (see `taylor_coeffs`).
    No gradient.
    """
    sg = jax.lax.stop_gradient
    lo, hi, nu = lim.lam_min, lim.lam_max, sg(lim.nu_max)
    m = 0.5 * (lo + hi)
    ok = jnp.isfinite(m) & (m > D2)
    near_l = (sg(hi - lo) < NEAR_SHELL * sg(hi)) & ok  # finite c_k only
    near_n = nu < NEAR_PLANAR * sg(D2)
    return m, near_l, near_n, ok


def taylor_coeffs(F: Fset, lim: Limits, iom: IOM | None = None) -> Taylor:
    """Coefficients of the Q-form Taylor regimes, from nested jvps of live F.

    c2..c4 are the tau-Taylor coefficients of F_lam at m = (lam_min+lam_max)/2
    (NaN where m is not inside the lambda branch); F2, F4 the u-Taylor
    coefficients of F_nu_u at u = 0. The flags carry no gradient; use_l is
    False where the c_k are NaN (m = D2: R below ~1e-8 Delta inside the foci).
    Near the foci (m - D2 small) c3 and c4 lose relative precision (F's
    sqrt(tau - D2) terms), but their share of Q there is small (MN, D2 = 4:
    ~2e-8 at R = 1e-3, falling as R^2).

    use_n (near-planar) needs nu_max < NEAR_PLANAR D2 and a series that
    converges over [0, nu_max]: F2, F4 finite and |F4| nu_max <
    NEAR_PLANAR_SERIES |F2|, so the dropped u^6 term is ~1e-9 of Q (measured
    6 (|F4| nu_max / |F2|)^2). That keeps smooth potentials on the series
    near the axis, where the direct quadrature loses up to ~1e-4 to
    cancellation, and keeps lanes whose nu-line runs through or near a cusp
    (F2, F4 up to ~1e100 there: NFW, MW2022's halo) on the direct quadrature.

    With ``iom`` (the live integrals F is built from) the near-circular
    lambda roots are Newton-polished first (`_polish`), and m and c2..c4
    follow the polished limits (returned as ``lam_min``, ``lam_max``).
    """
    m, use_l, near_n, ok = _candidates(lim, F.D2)
    c = _coeffs(F, jnp.where(ok, m, 2 * F.D2))
    c2, c3, c4 = (jnp.where(ok, c[k], jnp.nan) for k in range(3))
    if iom is not None:
        lim, near = _polish(F, iom, lim, m, ok, c)
        # m and c2, c3 at the polished midpoint: the quartic, shifted by dm
        # (dm <~ 0.1 w and w < 1e-3 lam: the dm^2 term is ~1e-14 of c2)
        m_p = 0.5 * (lim.lam_min + lim.lam_max)
        dm = jnp.where(near, m_p - m, 0.0)
        c2 = jnp.where(near, c2 + dm * (3 * c3 + 6 * dm * c4), c2)
        c3 = jnp.where(near, c3 + 4 * dm * c4, c3)
        m = jnp.where(near, m_p, m)
    sg = jax.lax.stop_gradient
    F2, F4 = sg(c[3]), sg(c[4])
    ok_n, ok_l = _series_match(F, lim, sg(m), sg(jnp.stack([c2, c3, c4])), F2, F4)
    use_n = (
        near_n
        & jnp.isfinite(F2)
        & jnp.isfinite(F4)
        & (jnp.abs(F4) * sg(lim.nu_max) < NEAR_PLANAR_SERIES * jnp.abs(F2))
        & ok_n
    )
    return Taylor(
        use_l & ok_l, m, c2, c3, c4, use_n, c[3], c[4], lim.lam_min, lim.lam_max
    )


def _polish(
    F: Fset, iom: IOM, lim: Limits, m: Array, ok: Array, c: Array
) -> tuple[Limits, Array]:
    """Near-circular lambda roots, Newton-polished on F's quartic at m.

    Where w = lam_max - lam_min < POLISH lam (not shell, snapped or pinned;
    ``ok``: m inside the lambda branch) `find_limits` is roundoff-limited:
    F's maximum is ~(w/lam)^2 of its terms. ``c`` (`_coeffs` at m) is F's
    quartic about m, its constant anchored on F(lam) from the momenta
    (exact and >= 0, as for the edge); N_POLISH Newton steps from the
    solver's roots, or from the roots of its quadratic part where those
    have the shorter Newton step, reach ~the ulp of lam.

    The quartic can miss F's root: a kink between m and the root, a kink
    just inside it (F is C1 at a hard cutoff, F'' jumps), or a lane near
    the foci past the quartic's convergence. So each polished root is kept
    only where it is on its own side of m, |F| there is within SNAP_NOISE
    of F's roundoff, and F' there is the quartic's to POLISH_SLOPE
    (`Fset.F_noise` and its jvp); it is otherwise the solver's. The
    simple-root tangent is reattached at the polished root with F' there.
    Returns (the limits, the polished-lane flag); the other lanes' limits
    are returned as given. Snapped lanes are not polished.
    """
    sg = jax.lax.stop_gradient
    lo0, hi0, m0, lam = sg(lim.lam_min), sg(lim.lam_max), sg(m), sg(iom.lam)
    f_mom = sg((iom.lam - iom.absnu) ** 2 * iom.lamdot**2 / (8 * iom.lam))
    a1, a2, a3, a4 = sg(c[5]), sg(c[0]), sg(c[1]), sg(c[2])
    xl = lam - m0
    a0 = f_mom - xl * (a1 + xl * (a2 + xl * (a3 + xl * a4)))

    def dp(x: Array) -> Array:
        return a1 + x * (2 * a2 + x * (3 * a3 + x * 4 * a4))

    L0 = jnp.stack([lo0, hi0])

    # Newton starts from the solver's root or from the root of the quartic's
    # quadratic part, whichever has the shorter Newton step on the quartic.
    # The solver's roots are roundoff-limited here and can sit well inside
    # the root pair, where 3 steps from near the vertex stop short (a
    # Hernquist lane with D2 = 1e4: 1.5% of the extent). Near the foci the
    # quadratic part is poor and the solver's root is the better start.
    def P(x: Array) -> Array:
        return a0 + x * (a1 + x * (a2 + x * (a3 + x * a4)))

    xs = L0 - m0
    xq = jnp.stack(_quad_roots(a0, a1, 2 * a2)[:2])
    use_q = jnp.abs(P(xq) / dp(xq)) < jnp.abs(P(xs) / dp(xs))  # NaN: False
    x = jnp.where(use_q, xq, xs)
    for _ in range(N_POLISH):
        x = x - P(x) / dp(x)
    L = m0 + x
    # F and F' at the polished roots: one jvp (F_noise is one phi call)
    (fL, nL), (dfL, _) = jax.jvp(F.F_noise, (L,), (jnp.ones_like(L),))
    fL, nL, dfL = sg(fL), sg(nL), sg(dfL)
    near = (
        (hi0 - lo0 < POLISH * lam)
        & ok  # c is F's quartic only inside the lambda branch
        & ~lim.shell
        & ~lim.snapped
        & ~lim.lmin_pinned
        & jnp.isfinite(L)
        # each on its own side of m: a step past lam (f_mom ~ 0 puts a root
        # there) converges to F's other root, which passes the checks below
        & (jnp.stack([-x[0], x[1]]) > 0)
        & (jnp.abs(fL) <= SNAP_NOISE * nL)  # a root of F itself; NaN fails
        & jnp.isfinite(dfL)  # inf <= inf would pass the slope check
        & (jnp.abs(dp(x) - dfL) <= POLISH_SLOPE * jnp.abs(dfL))  # and of P's slope
    )
    L = jnp.where(near, L, L0)
    t = -_over(_tang_at(F.F_lam, L, F.D2, 2 * F.D2), dfL)
    lim = lim._replace(
        lam_min=jnp.where(near[0], L[0] + t[0], lim.lam_min),
        lam_max=jnp.where(near[1], L[1] + t[1], lim.lam_max),
    )
    return lim, jnp.any(near)


def _series_match(
    F: Fset, lim: Limits, m: Array, c: Array, F2: Array, F4: Array
) -> tuple[Array, Array]:
    """A-posteriori check of the two series against F itself (no gradient).

    The coefficients are nested jvps at a point, which cannot see a kink:
    d|z|/dz = 0 at z = 0 hides an |u| term (a razor-thin disk), and a jump in
    F'' inside the orbit (a hard cutoff) is invisible from m. So F is
    evaluated at u = sqrt(nu_max)/2 (with F(0)) and at m -+ w/4, and each
    series must match it to SERIES_CHECK_N (near-planar, of F(u_h) - F(0))
    or SERIES_CHECK_L (near-shell, of F) plus SNAP_NOISE x F's roundoff there
    (`Fset.F_noise`); NaN fails. Smooth potentials pass and a resolvable
    kink fails by orders of magnitude; a weak kink can pass. The roundoff
    term takes |Phi| as the size of Phi's terms (see `_noise_snap`): where
    they cancel, use_n can change with the batch's chunking and with jit.
    """
    sg = jax.lax.stop_gradient
    lo, hi = sg(lim.lam_min), sg(lim.lam_max)
    w = hi - lo
    uh = 0.5 * jnp.sqrt(jnp.where(lim.nu_max > 0, sg(lim.nu_max), 0.0))
    x = jnp.stack([-0.25 * w, 0.25 * w])
    tau = jnp.concat([jnp.stack([jnp.zeros_like(uh), uh * uh]), m + x])
    f, noise = (sg(a) for a in F.F_noise(tau))
    # near-planar: F(u_h) - F(0) against F2 u_h^2 + F4 u_h^4
    df = f[1] - f[0]
    err_n = jnp.abs(df - (F2 + F4 * uh * uh) * uh * uh)
    ok_n = err_n <= SERIES_CHECK_N * jnp.abs(df) + SNAP_NOISE * (noise[0] + noise[1])
    # near-shell: F(m + x) against -Q(x) (tau - lo)(hi - tau)
    q = c[0] + c[1] * x + c[2] * (x * x + w * w / 4)
    err_l = jnp.abs(f[2:] + q * (m + x - lo) * (hi - m - x))
    ok_l = jnp.all(err_l <= SERIES_CHECK_L * jnp.abs(f[2:]) + SNAP_NOISE * noise[2:])
    return ok_n, ok_l


def _coeffs(F: Fset, x: Array) -> Array:
    """(c2, c3, c4) of F_lam at x, (F2, F4) of F_nu_u at u = 0, F_lam'(x).

    Both jvp nests are written in this one function with one stacked
    output, and plain autodiff runs through it. Consumers must read the
    coefficients by indexing (the `Taylor` fields): XLA then fuses the two
    nests' value-and-gradient work. Summing the stacked output, or calling
    the two nests separately, compiles to far more flops under ``vmap``
    (pinned by a flop-count test).
    """
    _, d1, d2, d3, d4 = _derivs(F.F_lam, x, 4)
    _, _, u2, _, u4 = _derivs(F.F_nu_u, jnp.zeros_like(x), 4)
    return jnp.stack([d2 / 2, d3 / 6, d4 / 24, u2 / 2, u4 / 24, d1])


def taylor_coeffs_batch(
    phi: Callable[[Array, Array], Array],
    w6: Array,
    D2: Array,
    lim: Limits,
    size: int | None = None,
    *,
    shell: bool = False,
) -> Taylor:
    """`taylor_coeffs` on a batch, computed only on the lanes that may use it.

    ``w6`` is (N, 6), ``D2`` and the ``lim`` fields (N,): the batch is the
    leading axis, and the live F of each lane is rebuilt here from
    (``phi``, ``w6``, ``D2``), as the pipeline builds it. The candidates
    (near-shell, or nu_max < NEAR_PLANAR D2) are gathered in order into
    chunks of ``size`` lanes (default max(16, N // 16)); one `jax.lax.scan`
    runs over the chunks, and a `jax.lax.cond` per chunk computes it only if
    it holds a candidate. The coefficients are scattered back. The cost so
    follows the number of candidates, with no cliff, and the nest is
    compiled once. The result on a candidate
    lane is `taylor_coeffs`' value and derivative; on the other lanes the
    coefficients are NaN and the flags False (m is always set).

    The limits are final only here: ``Taylor.lam_min`` and ``lam_max``
    (``m`` their midpoint). On the candidates the near-circular lambda roots
    are Newton-polished (`_polish`, w < POLISH lam), at O(candidates) cost;
    the other lanes return the input limits bitwise (values and both
    derivative modes).

    With ``shell=True``, ``lim`` must come from ``attach_tangents(...,
    shell=False)`` (default limits would get the shell tangent twice): the
    shell lanes' double-root tangent (shell lanes are candidates) is added
    to their limits here, before the coefficients.

    Under an outer ``vmap`` the conds become selects and every chunk runs:
    correct, but without the saving. Call it on the flat batch.
    """
    N = w6.shape[0]
    K = max(16, N // 16) if size is None else size
    n_ch = -(-N // K)
    m, near_l, near_n, _ = jax.vmap(_candidates)(lim, D2)
    cand = near_l | near_n
    n_cand = jnp.sum(cand)

    def nan_taylor(n: int) -> Taylor:
        nans, false = jnp.full(n, jnp.nan, m.dtype), jnp.zeros(n, bool)
        return Taylor(false, nans, nans, nans, nans, false, nans, nans, nans, nans)

    # the other lanes: the input limits and their midpoint
    base = nan_taylor(N)._replace(m=m, lam_min=lim.lam_min, lam_max=lim.lam_max)
    if N == 0:  # static: no chunk to gather into
        return base

    # Rematerialised in reverse mode: the cond's linearisation otherwise keeps
    # its branches' residuals of the 4th-order nest.
    @jax.checkpoint
    def one(w: Array, d2: Array, lim1: Limits) -> Taylor:
        iom = integrals_of_motion(phi, w, d2)
        F = make_F(phi, iom)
        if shell:
            dsh = _double_tangent(F, F, lim1)  # F'' is stop-gradiented there
            lim1 = lim1._replace(lam_min=lim1.lam_min + dsh, lam_max=lim1.lam_max + dsh)
        return taylor_coeffs(F, lim1, iom)

    idx = jnp.nonzero(cand, size=n_ch * K, fill_value=N)[0].reshape(n_ch, K)

    def chunk(_: None, xs: tuple[Array, Array]) -> tuple[None, Taylor]:
        k, ix = xs

        def take(x: Array) -> Array:
            return x.at[ix].get(mode="clip")

        def run(_: None) -> Taylor:
            t: Taylor = jax.vmap(one)(take(w6), take(D2), jax.tree.map(take, lim))
            return t

        empty = nan_taylor(K)
        return None, jax.lax.cond(k * K < n_cand, run, lambda _: empty, None)

    _, ts = jax.lax.scan(chunk, None, (jnp.arange(n_ch), idx))
    flat = idx.reshape(-1)
    return Taylor._make(
        a.at[flat].set(x.reshape(-1), mode="drop")
        for a, x in zip(base, ts, strict=True)
    )

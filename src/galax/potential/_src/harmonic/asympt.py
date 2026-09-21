r"""Asymptotic power-law continuation of the multipole coefficient profiles.

Outside ``[r_min, r_max]`` the cubic Hermite spline of `spline` has nothing to
say: continuing its edge cubic in :math:`\log r` is unbounded and wrong by
construction. Close to the boundary it is respectable -- it reads the same
knot derivatives the tail does -- but it degrades without limit. On a
Hernquist monopole over ``[0.05, 20]`` the edge cubic's relative error runs
:math:`9.2\times 10^{-3}`, :math:`0.13`, :math:`1.43` at 2, 5 and
:math:`10 r_\max`, against :math:`1.0\times 10^{-2}`, :math:`1.8\times 10^{-2}`,
:math:`2.3\times 10^{-2}` for the continuation, and somewhere in that range it
changes sign -- a repulsive force from a bound system.
(`test_the_edge_cubic_really_does_change_sign` pins that; the table is
`test_end_to_end_beats_the_edge_cubic`.) This module replaces it with the
physical continuation,

.. math::

    \Phi_{lm}(r) = W (r/r_1)^v + U (r/r_1)^s + Q (r/r_1)^2 ,

following ``agama``'s ``computeExtrapolationCoefs`` / ``initAsympt`` /
``PowerLawMultipole`` in ``src/potential_multipole.cpp``. Here :math:`r_1` is
the boundary node, :math:`v = l` inward and :math:`v = -l-1` outward -- the
two source-free solid-harmonic solutions, so :math:`W (r/r_1)^v` is what
survives when there is no mass beyond the boundary -- :math:`s` is set by the
asymptotic power-law slope of the density, :math:`\rho \sim r^{s-2}`, and
:math:`Q (r/r_1)^2` is the uniform-density piece.

Why it lives here and not in `spline`
-------------------------------------
The continuation needs :math:`l`, and a spline evaluator has no business
knowing about spherical harmonics: `eval_log_spline` is a generic
knots-and-derivatives evaluator, reused and tested as such. It also needs a
root find, which must happen *once at build time* -- `eval_log_spline` is
called from inside a `diffrax` scan body, so folding a root find into it would
re-solve on every integration step. Hence two functions: `asymptotic_coeffs`
fits the continuation at build time, `eval_log_spline_asympt` evaluates the
spline and its two tails. Inside the knots the latter delegates to
`eval_log_spline` unchanged, so interior values stay bit-identical.

Where this departs from ``agama``
---------------------------------
* The degenerate :math:`s = v` case -- ``agama``'s
  :math:`W (r/r_1)^v + U (r/r_1)^v \ln(r/r_1)`, which is the *generic* outer
  monopole of an :math:`r^{-3}` halo such as NFW -- is not a separate branch
  here. Writing the two power laws as the divided difference

  .. math::

      \frac{x^s - x^v}{s - v} \xrightarrow{s \to v} x^v \ln x

  makes the limit analytic, so one expression covers both cases and there is
  no branch to get wrong and no cancellation near the degeneracy. ``agama``
  instead snaps :math:`s` to :math:`v` inside
  :math:`|A+1| < \sqrt{\epsilon}` precisely to dodge that cancellation.
* :math:`s` is found by bisection on a residual that is provably monotone
  (see `_slope_and_scale`) rather than through the Lambert :math:`W` function.
* The coefficient of the :math:`x^s` term is rewritten so that the boundary
  value *and* derivative are reproduced to round-off by construction rather
  than as an outcome of the algebra; see `asymptotic_coeffs`.
"""

__all__: tuple[str, ...] = ()

import functools as ft

from collections.abc import Callable
from jaxtyping import Array, Bool, Float

import jax

import quaxed.numpy as jnp

from .spline import eval_log_spline

_Fn = Callable[[Float[Array, "*s"]], Float[Array, "*s"]]

_SERIES_TOL: float = 1e-2
r"""Switch to the series form of :math:`(e^z-1)/z` below this ``|z|``.

Needed at all because the quotient is :math:`0/0` at :math:`z = 0` and so not
differentiable there. Away from zero the quotient loses about
:math:`\epsilon/|z|` to cancellation -- in the value and in the derivatives
alike, since differentiating
:math:`(e^{aL} - e^{bL})/(a-b)` gives :math:`(a^2 e^{aL} - b^2 e^{bL})/(a-b)`,
the same single subtraction rather than a worse one. Measured against a
50-digit reference, the second derivative (which `hessian` and
`tidal_tensor` read) comes back at

======== ================= ==============
``|z|``  quotient rel. err :math:`\epsilon/|z|`
======== ================= ==============
1e-2     1.3e-14           2.2e-14
1e-4     2.9e-12           2.2e-12
1e-6     2.1e-11           2.2e-10
======== ================= ==============

so the seam only has to be far enough from zero that :math:`\epsilon/|z|` is
still small. At :math:`10^{-2}` the quotient side costs ~1e-14 relative and
the series side, truncating at :math:`z^7/40320`, costs ~1e-16 -- both
comfortably inside the quadrature error of anything that calls this.
"""

_SAFETY: float = 100.0
"""Round-off guard, in units of the working epsilon. Matches ``agama``."""

_LN_HUGE: float = 700.0
r"""Largest ``|exponent * ln x|`` allowed into `jnp.exp`.

Just under ``log(DBL_MAX) = 709.78``. `eval_log_spline_asympt` clamps
:math:`L` so no term can reach an ``inf`` -- and so that :math:`r = 0`, which
arrives as :math:`L = -\infty`, cannot turn the monopole's :math:`v = 0`
into an indeterminate :math:`0 \times \infty`.
"""

_N_BISECT: int = 60
"""Bisection steps. A bracket of width 9 shrinks to ``8e-18``, below eps."""

_S_INNER: tuple[float, float, float] = (-1.0, 8.0, 2.0)
r"""Bracket for the inward slope.

The lower end is ``agama``'s convergence guard: the density may not grow
faster than :math:`r^{-3}` as :math:`r \to 0`, i.e. :math:`s - 2 \ge -3`, or
the enclosed mass diverges. The upper end is ``agama``'s "if :math:`s` is too
high the parameters cannot be reliably determined anyway". The third entry
is the fallback slope used when the boundary data does not determine one --
``agama``'s own, a constant-density core.
"""

_S_OUTER: tuple[float, float, float] = (-8.0, 0.0, -2.0)
r"""Bracket for the outward slope.

The upper end is ``agama``'s convergence guard: the density must fall faster
than :math:`r^{-2}`, i.e. :math:`s - 2 < -2`, or the enclosed mass diverges.
Both ends keep :math:`(r/r_\max)^s` bounded outward, so a bracket that binds
degrades to a bounded tail rather than a divergent one. The third entry is
the fallback slope -- ``agama``'s, an :math:`r^{-4}` falloff.
"""

_Q_BRACKET: tuple[float, float] = (2.0 + 3.0 * 1.4901161193847656e-08, 8.0)
r"""Bracket for the four-parameter inner-monopole slope, from ``agama``.

:math:`s = 2` is always a root of that residual -- it is the :math:`Q` term
itself -- so the lower end is nudged off it by :math:`3\sqrt{\epsilon}`.
"""


def _pow_diff(
    a: Float[Array, "..."], b: Float[Array, "..."], ln_x: Float[Array, "..."], /
) -> Float[Array, "..."]:
    r"""Return :math:`(x^a - x^b)/(a-b)` for :math:`x = e^{\ln x}`.

    The :math:`a \to b` limit is :math:`x^b \ln x`, which is what makes the
    degenerate :math:`s = v` case of the continuation an ordinary point rather
    than a branch. Near it the quotient is :math:`0/0`, so the series form of
    :math:`(e^z-1)/z` is used instead; both branches are finite everywhere, so
    the unselected one cannot poison a gradient with a `nan`.
    """
    z = (a - b) * ln_x
    small = jnp.abs(z) < _SERIES_TOL
    series = 1.0 + z * (
        0.5 + z * (1 / 6 + z * (1 / 24 + z * (1 / 120 + z * (1 / 720 + z / 5040))))
    )
    return jnp.where(  # type: ignore[no-any-return]
        small,
        jnp.exp(b * ln_x) * ln_x * series,
        (jnp.exp(a * ln_x) - jnp.exp(b * ln_x)) / jnp.where(small, 1.0, a - b),
    )


def _bisect(
    fn: _Fn,
    lo: Float[Array, "*s"],
    hi: Float[Array, "*s"],
    increasing: Bool[Array, "*s"],
    /,
) -> Float[Array, "*s"]:
    """Bisect ``fn`` on ``[lo, hi]``, elementwise, keeping the root's side.

    A fixed `jax.lax.fori_loop` rather than an `optimistix` solve. These
    residuals are one-dimensional and -- for the three-parameter fit --
    provably monotone, so there is nothing to converge-check and no
    bracketing to fail; and when the root lies *outside* the physically
    admissible bracket this loop walks to the nearest endpoint, which is
    exactly the clamping the slope guards call for. `optimistix` would instead
    report that case as a solver failure to be post-processed into the same
    clamp. ``increasing`` says which way ``fn`` runs, so the walk goes the
    right way when there is no sign change to preserve; a generic
    sign-preserving bisection walks to the *far* endpoint instead, which is
    how a root sitting exactly on the bracket edge -- ``s = -1`` for a
    point-mass monopole, say -- gets lost.
    """

    def body(_: Array, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        right = (fn(mid) < 0.0) == increasing
        return jnp.where(right, mid, lo), jnp.where(right, hi, mid)

    lo, hi = jax.lax.fori_loop(0, _N_BISECT, body, (lo, hi))
    return 0.5 * (lo + hi)


def _polish(
    fn: _Fn, s: Float[Array, "*s"], bracketed: Bool[Array, "*s"], /
) -> Float[Array, "*s"]:
    r"""Re-attach the implicit derivative to a detached bisection root.

    Bisection is a tree of comparisons and carries no derivative information:
    differentiating it gives zero. The correction below is *identically zero*
    in value -- ``r - stop_gradient(r)`` -- so the bisection result is
    returned bit-for-bit, while its derivative is
    :math:`-(\partial R/\partial\theta)/(\partial R/\partial s)`, exactly
    the implicit-function derivative an `optimistix` root find would supply.
    Three lines and no second solve. Writing it value-neutrally, rather than
    as a plain Newton step, keeps the returned root exactly the one the
    bracketing established: a real step would move it by the residual's own
    noise floor, which is not an improvement once bisection has already
    converged, and would leave the reported root depending on the arithmetic
    of the correction rather than on the sign changes that found it.

    ``bracketed`` says whether the interval actually holds a sign change.
    Where it does not, the returned value is a clamped endpoint whose true
    sensitivity to the data is zero, and re-attaching an implicit derivative
    there would invent one -- so the term is dropped.
    """
    s = jax.lax.stop_gradient(s)
    r, dr = jax.jvp(fn, (s,), (jnp.ones_like(s),))
    dr = jax.lax.stop_gradient(dr)
    ok = bracketed & (dr != 0.0)
    step = (r - jax.lax.stop_gradient(r)) / jnp.where(ok, dr, 1.0)
    return s - jnp.where(ok, step, 0.0)  # type: ignore[no-any-return]


def _solve(
    fn: _Fn,
    bracket: tuple[float, float],
    like: Float[Array, "*s"],
    increasing: Bool[Array, "*s"] | None = None,
    /,
) -> Float[Array, "*s"]:
    """Bracketed root of ``fn``, clamped to ``bracket``, elementwise."""
    lo, hi = (jnp.zeros_like(like) + b for b in bracket)
    f_lo, f_hi = fn(lo), fn(hi)
    up = (f_lo < 0.0) if increasing is None else increasing
    return _polish(fn, _bisect(fn, lo, hi, up), f_lo * f_hi <= 0.0)


def _slope(
    P1: Float[Array, "*s"],
    D1: Float[Array, "*s"],
    P2: Float[Array, "*s"],
    h: Float[Array, ""],
    v: Float[Array, "*s"],
    bracket: tuple[float, float, float],
    eps: Float[Array, ""],
    /,
) -> Float[Array, "*s"]:
    r"""Fit the power-law slope :math:`s` from three boundary values.

    Writing :math:`x = r/r_1` and :math:`L = \ln x`, the three-parameter form
    :math:`W x^v + U x^s` that reproduces :math:`\Phi(r_1) = P_1` and
    :math:`d\Phi/d\ln r\,(r_1) = D_1` is, with :math:`K = D_1 - v P_1`,

    .. math::

        \Phi(L) = P_1 x^v + K \frac{x^s - x^v}{s - v} ,

    since the second term and its derivative are :math:`0` and :math:`K` at
    :math:`L = 0`. Both boundary constraints therefore hold *identically* in
    :math:`s` -- which is what makes the join C1 to round-off, whatever this
    function returns. The remaining constraint, :math:`\Phi(h) = P_2` with
    :math:`h = \ln(r_2/r_1)`, is what fixes :math:`s`.

    That residual is strictly monotone in :math:`s`: the divided difference
    equals :math:`x^v h\,g((s-v)h)` with :math:`g(z) = (e^z-1)/z`, so its
    :math:`s`-derivative is :math:`K x^v h^2 g'(z)` and runs the way
    :math:`K` does. The root is unique and bisection cannot fail.

    Where the boundary data does not determine a slope at all, the fallback
    slope of ``bracket`` is returned. ``agama`` instead zeroes :math:`U`
    there, leaving the bare solid harmonic; that reproduces :math:`P_1` but
    *not* :math:`D_1`, so it would put a step in the force at the join. Note
    the amplitude :math:`K` needs no such rescue: where the data is pure
    cancellation noise :math:`K` is itself ~zero, so the tail degenerates to
    the solid harmonic on its own, continuously.
    """
    lo, hi, fallback = bracket
    K = D1 - v * P1
    E = jnp.exp(v * h)
    # Grouped so the two large, nearly equal knot values cancel against each
    # other first; `P1 * E + K * pdiff - P2` mixes scales and then cancels.
    gap = P1 * E - P2

    def residual(s: Float[Array, "*s"]) -> Float[Array, "*s"]:
        return gap + K * _pow_diff(s, v, h)  # type: ignore[no-any-return]

    s = _solve(residual, (lo, hi), P1, K > 0.0)

    # `agama`'s guards. A root exists only if `h E / T > 0` with T the target
    # divided difference, equivalently `h K gap < 0`; that is `agama`'s
    # `A < 0` test rewritten without the division. The other two reject a
    # numerator or denominator that is pure cancellation noise.
    tol = _SAFETY * eps
    good = (
        (h * K * gap < 0.0)
        & (jnp.abs(K) >= tol * jnp.maximum(jnp.abs(D1), jnp.abs(v * P1)))
        & (jnp.abs(gap) >= tol * jnp.maximum(jnp.abs(P1 * E), jnp.abs(P2)))
    )
    return jnp.where(good, s, fallback)  # type: ignore[no-any-return]


def _inner_monopole_q(
    P1: Float[Array, "*s"],
    D1: Float[Array, "*s"],
    P2: Float[Array, "*s"],
    D2: Float[Array, "*s"],
    h: Float[Array, ""],
    /,
) -> tuple[Float[Array, "*s"], Float[Array, "*s"], Bool[Array, "*s"]]:
    r"""Four-parameter inward monopole fit, giving the :math:`Q x^2` term.

    .. warning::

        Opt-in only, via ``asymptotic_coeffs(..., cored_monopole=True)``, and
        **not** safe on spline-derived boundary derivatives. Two properties
        conspire against it there:

        1. :math:`s = 2` is an *exact* root of the residual for arbitrary
           data -- identically zero, not merely small. `_Q_BRACKET` nudges
           its endpoint by :math:`3\sqrt{\epsilon}` to avoid it; that is not
           enough once (2) applies.
        2. The acceptance gate compares a residual that is
           :math:`O(h^4)` against a scale that is :math:`O(1)`, so as the
           grid refines the whole bracket eventually passes and bisection
           drops onto whichever endpoint it walked to.

        The result is that refinement makes the answer *worse*. On an NFW
        monopole over ``[0.05, 20]`` with closed-form knots, the relative
        error at :math:`r_\min/2`:

        ======  =========  =============  ==========
        ``n_r``  ``s``      ``|Q|/|P1|``  rel. error
        ======  =========  =============  ==========
        1024     0.96734    0             1.1e-5
        2048     0.96738    0             1.1e-5
        4096     8.00000    7.2e-3        1.0e-3
        8192     2.00000    7.2e+03       3.0e-4
        ======  =========  =============  ==========

        -- a 92x degradation from one refinement, non-monotone, with the fit
        pinned at a bracket endpoint rather than an interior root. Across 147
        realistic monopole builds the fit was accepted 41 times and *never
        once* with :math:`s` strictly interior.

        Requiring a sign change in the bracket does not rescue it, because
        :math:`s = 2` is a genuine root. A gate that would work has to be
        dimensionless -- the residual small relative to its own range over
        the bracket, not to ``scale`` -- and must additionally bound
        :math:`|den|` away from zero so :math:`U` and :math:`Q` stay
        conditioned. Until then the three-parameter form is used, which
        converged monotonically in every configuration tested.

        Contrast `_slope`, whose gates are relative and whose residual is
        provably monotone with a unique root; it needs no such caveat.

    Only the inward monopole carries :math:`Q`: for :math:`v = 0` the
    continuation can afford a fourth parameter, fitting
    :math:`W + U x^s + Q x^2` to the value *and* derivative at both boundary
    nodes, which resolves a density :math:`\rho \sim r^{s-2} + \text{const}`
    -- a cored profile plus a rising correction -- rather than a single power
    law. Every other mode uses the three-parameter form. This is ``agama``'s
    ``v==0`` branch of ``computeExtrapolationCoefs``, residual and all.

    Returns ``(s, Q, accept)``. ``accept`` is `False` where the fit is not
    usable, in which case the caller falls back on the three-parameter form,
    matching ``agama``'s "only accept this extrapolation if the density has
    the same sign at ``r=0`` and ``r=r1``".
    """
    # `agama` works in dPhi/dr; ours are d/dln r, so r1*dPhi1 = D1 and
    # r1*dPhi2 = D2/ratio. The residual is `agama`'s, scaled by r1.
    ratio = jnp.exp(h)
    d2 = D2 / ratio

    def residual(s: Float[Array, "*s"]) -> Float[Array, "*s"]:
        rsm1 = jnp.exp((s - 1.0) * h)
        return (  # type: ignore[no-any-return]
            (P2 - P1) * (rsm1 - ratio)
            + (d2 - D1 * rsm1) * 0.5 * (ratio**2 - 1.0)
            + (D1 * ratio - d2) * (ratio * rsm1 - 1.0) / s
        )

    s = _solve(residual, _Q_BRACKET, P1)

    rsm1 = jnp.exp((s - 1.0) * h)
    den = rsm1 - ratio
    ok = den != 0.0
    safe = jnp.where(ok, den, 1.0)
    U = (d2 - D1 * ratio) / safe / s
    Q = (D1 * rsm1 - d2) / safe * 0.5

    # `agama`: accept only if the implied density has one sign over [0, r1].
    # The residual check is ours: the four-parameter residual is not monotone,
    # so bisection is not guaranteed to have found a root at all.
    scale = jnp.abs(P2 - P1) + jnp.abs(D1) + jnp.abs(d2)
    accept = (
        ok
        & jnp.isfinite(s)
        & jnp.isfinite(U)
        & jnp.isfinite(Q)
        & (6.0 * Q * (6.0 * Q + s * (s + 1.0) * U) > 0.0)
        & (jnp.abs(residual(s)) < 1e-10 * scale)
    )
    return s, Q, accept


@ft.partial(jax.jit, static_argnames=("cored_monopole",))
def asymptotic_coeffs(
    log_r: Float[Array, "n_r"],
    values: Float[Array, "n_r *rest"],
    derivs: Float[Array, "n_r *rest"],
    l_per_mode: Float[Array, "*rest"],
    /,
    *,
    cored_monopole: bool = False,
) -> Float[Array, "2 rows *rest"]:
    r"""Fit the inward and outward power-law continuations of each mode.

    ``values`` and ``derivs`` are the knot values and :math:`\log r`
    derivatives of `fit_log_spline`; only the two knots at each end are used.
    ``l_per_mode`` carries :math:`l` as a float per trailing entry, matching
    `solve_poisson_lm`.

    ``cored_monopole`` opts in to the four-parameter inward monopole fit, the
    one that carries the :math:`Q x^2` term. It is **off by default and must
    stay off for spline-derived derivatives**: see `_inner_monopole_q` for the
    measurements. Pass it only with boundary derivatives known to be accurate
    to better than the fit's own :math:`O(h^4)` residual -- analytic ones, in
    practice. With it off the three-parameter form is used -- there is no
    ``Q`` row at all, rather than a zero one -- and it converges monotonically
    in every configuration tested.

    Returns ``(v, s, B)`` per side -- or ``(v, s, B, Q)`` when
    ``cored_monopole`` is set -- to be handed to `eval_log_spline_asympt`,
    which reads the presence of the ``Q`` row off the shape. The tail is
    evaluated as

    .. math::

        \Phi(L) = P_1 x^v + B \frac{x^s - x^v}{s-v} + Q (x^2 - x^v) ,
        \qquad x = e^L = r/r_1 ,

    with :math:`P_1` read back from ``values``. Written this way the boundary
    value is :math:`P_1` bit-for-bit and the boundary derivative is
    :math:`v P_1 + B + (2-v) Q`, so choosing ``B = D1 - v P1 - (2-v) Q``
    pins the join to the spline in value *and* slope to round-off, whatever
    the fit does. ``agama`` instead lets both follow from its algebra, which
    is C1 only to the accuracy of that algebra.
    """
    eps = jnp.finfo(values.dtype).eps
    shape = values.shape[1:]
    l = jnp.broadcast_to(l_per_mode, shape)

    sides = []
    for i1, i2, v, bracket in (
        (0, 1, l, _S_INNER),
        (-1, -2, -l - 1.0, _S_OUTER),
    ):
        h = log_r[i2] - log_r[i1]
        P1, D1, P2, D2 = values[i1], derivs[i1], values[i2], derivs[i2]
        s = _slope(P1, D1, P2, h, v, bracket, eps)
        sides.append((h, P1, D1, P2, D2, v, s))

    # Cross-l safeguard, inward only. Inward `_S_INNER` reaches -1, so a
    # fitted `s` can make an l > 0 mode *diverge* toward the centre faster
    # than the monopole does; clipping it to the monopole's slope stops that,
    # and it measurably earns its keep (on a flattened-NFW (4,2) mode whose
    # fit had walked to the bracket edge it cuts the error 1.74 -> 0.42).
    #
    # Outward the same clip is applied nowhere, because there it is both
    # unnecessary and harmful. `_S_OUTER` caps `s` at 0, so outward no mode
    # can diverge; and `s0` is the slope of the monopole's *correction* term,
    # not of its leading `x^-1`, so comparing against it has no physical
    # content. What the fitted `s` is actually reporting outward is that the
    # density does not stop at `r_max` -- the grid is truncated, the mass is
    # not -- so a decay slower than the solid harmonic is correct. Measured
    # on a q = 0.6 flattened NFW at l_max = 4 against a build on a 20x wider
    # grid, clipping costs a factor of 3-5 on exactly the modes it binds:
    #
    #     mode    s clipped   s unclipped    rel. err at r = 1500 kpc
    #     (2,0)     -3.2488       -1.2795       0.855  ->  0.271
    #     (4,0)     -3.2488       -1.1502       0.941  ->  0.192
    #
    # Clipping to `v` instead of `s0` was also measured, and is no better
    # than clipping to `s0` (0.833 and 0.988 on those two modes).
    is_mono = l == 0
    has_mono = jnp.any(is_mono)
    n_mono = jnp.maximum(jnp.sum(is_mono), 1.0)
    out = []
    for j, (h, P1, D1, P2, D2, v, fitted) in enumerate(sides):
        if j == 0:
            # With no monopole in the set there is nothing to clip against:
            # `s0` would be a fictitious 0, not any mode's slope.
            s0 = jnp.sum(jnp.where(is_mono, fitted, 0.0)) / n_mono
            s = jnp.where(is_mono | ~has_mono, fitted, jnp.maximum(fitted, s0))
        else:
            s = fitted
        Q = jnp.zeros_like(s)
        if j == 0 and cored_monopole:  # inward monopole only, and opt-in
            s_q, q, accept = _inner_monopole_q(P1, D1, P2, D2, h)
            use = accept & is_mono
            s, Q = jnp.where(use, s_q, s), jnp.where(use, q, Q)
        # The amplitude follows from the C1 constraint itself, never from
        # the fit, so the join holds whatever `s` and `Q` came out as.
        B = D1 - v * P1 - (2.0 - v) * Q
        # The `Q` row is carried only when it can be non-zero. Its presence
        # is what tells `eval_log_spline_asympt` whether to evaluate the
        # term, so the two cannot disagree: a caller who opts in gets four
        # rows and the term is honoured, and one who does not cannot
        # accidentally pay for an `exp` against a structural zero.
        out.append(jnp.stack([v, s, B, Q] if cored_monopole else [v, s, B]))
    return jnp.stack(out)  # type: ignore[no-any-return]


def eval_log_spline_asympt(
    log_r: Float[Array, "n_r"],
    values: Float[Array, "n_r *rest"],
    derivs: Float[Array, "n_r *rest"],
    coefs: Float[Array, "2 rows *rest"],
    log_rq: Float[Array, "*batch"],
    /,
) -> Array:
    r"""Evaluate the splined modes with power-law continuation outside the knots.

    Inside ``[log_r[0], log_r[-1]]`` this is `eval_log_spline` unchanged and
    bit-identical; outside it is the asymptotic form fitted by
    `asymptotic_coeffs`. ``coefs`` must come from the same ``values`` and
    ``derivs``.

    Both tails are evaluated on a clamped :math:`L`, so every branch of the
    select is finite for every query radius and no `nan` leaks into a
    gradient. The clamp bounds :math:`L` in *magnitude*, not only in sign:
    bounding the sign alone still lets :math:`r = 0` reach the tail as
    :math:`L = -\infty`, where the monopole's :math:`v = 0` makes
    :math:`v L` an indeterminate :math:`0 \times \infty` and the whole mode
    comes back `nan`.

    What the clamp guarantees is finiteness, not fidelity. It is set by the
    *largest* exponent in the mode, so once it binds it also truncates the
    smaller ones -- which outward is the slowest-decaying, dominant term --
    and the result goes to a constant plateau instead of continuing to decay.
    That only happens beyond :math:`|L| \ge 700/\max(|v|, |s|)`, i.e. past
    :math:`r/r_\max \sim 10^{23}` at :math:`l_\max = 12`, so nothing
    physical reaches it; inside that range the clamped and exact results
    agree.

    At :math:`r = 0` the returned value is the *monopole's* limit
    :math:`W = P_1 - B/s - Q` when :math:`s > 0` (for :math:`l > 0`,
    :math:`v = l > 0` so every term carries :math:`x^v \to 0` and the limit
    is simply 0). When :math:`s < 0` -- a
    Kepler monopole, say -- :math:`x^s` diverges and the true limit is
    :math:`-\infty`; the clamp returns a large finite number instead, whose
    magnitude is an artifact of the bound and whose gradient is 0 rather than
    infinite. Finite and wrong beats `nan` for an integrator that strays
    there, but it is not the limit.

    The spline branch is likewise evaluated on a clamped ``log_rq``, which
    also keeps the edge cubic from overflowing far outside the grid.
    """
    trailing = (1,) * (values.ndim - 1)

    def rs(a: Array) -> Array:
        return a.reshape((*a.shape, *trailing))

    def tail(P1: Array, side: Array, ln_x: Array, *, q_term: bool) -> Array:
        """One side's continuation. ``q_term`` is static, so it is free."""
        v, s, B = side[0], side[1], side[2]
        # Keep every exponent below the `exp` overflow threshold, so no term
        # is an inf or a `0 * inf`.
        largest = jnp.maximum(jnp.abs(v), jnp.abs(s))
        if q_term:
            largest = jnp.maximum(largest, 2.0)
        ln_x = jnp.clip(ln_x, -_LN_HUGE / largest, _LN_HUGE / largest)

        out = P1 * jnp.exp(v * ln_x) + B * _pow_diff(s, v, ln_x)
        if q_term:
            out = out + side[3] * (jnp.exp(2.0 * ln_x) - jnp.exp(v * ln_x))
        return out  # type: ignore[no-any-return]

    core = eval_log_spline(log_r, values, derivs, jnp.clip(log_rq, log_r[0], log_r[-1]))
    # Q is fitted on the inward side only, and only when asked for: four rows
    # means it may be live, three that it is structurally absent. Either way
    # the outward tail omits the term rather than multiplying by a zero,
    # which saves an `exp` per mode per point. The shape is static, so the
    # decision is free, and the two functions cannot disagree about it.
    has_q = coefs.shape[1] == 4
    inner = tail(
        values[0], coefs[0], rs(jnp.minimum(log_rq - log_r[0], 0.0)), q_term=has_q
    )
    outer = tail(
        values[-1], coefs[1], rs(jnp.maximum(log_rq - log_r[-1], 0.0)), q_term=False
    )
    return jnp.where(  # type: ignore[no-any-return]
        rs(log_rq < log_r[0]), inner, jnp.where(rs(log_rq > log_r[-1]), outer, core)
    )

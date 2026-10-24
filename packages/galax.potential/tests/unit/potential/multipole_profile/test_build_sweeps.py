"""Parameter sweeps over `build_expansion`'s float32 accuracy.

Split from `test_build.py`: these are the slow tests, one build per row, and
`pytest-xdist` runs a file as one unit (``--dist=loadfile``), so keeping them
beside the fast ones made that file the longest chain in CI.

Each row would compile its own build, because `build_expansion` is jitted with
the density static and every row used to close over its own parameters. The
swept numbers are carried as traced values instead, so rows that share a grid
size share a compile; the numerics are unchanged to round-off.
"""

from functools import partial

import jax
import pytest

import quaxed.numpy as jnp

from galax.potential._src.builtin.multipole_profile.build import build_expansion
from galax.potential._src.harmonic import default_angular_resolution, lm_keys
from galax.potential._src.utils import safe_vector_norm

_F32_EPS = float(jnp.finfo(jnp.float32).eps)
"""Round-off floor for the accuracy bounds; see the assertions that use it."""


def _dehnen_slope_in_t(xyz, t):
    """``1 / (r**t (1 + r)**(4 - t))``: the slope is the *traced* ``t``.

    `build_expansion` is jitted with its density static, so a fresh closure per
    row compiles once per row. Carrying the one swept number in ``t`` makes
    the rows that share a grid size share a compile.
    """
    r = safe_vector_norm(xyz)
    return 1.0 / (r**t * (1.0 + r) ** (4.0 - t))


# Measured on `main` (an unanchored solve) at 7260498c, float32, max relative
# error against the Dehnen closed form, keyed by (r_min, r_max, gamma, n_r).
# Anchoring must never do worse than these, and should usually do far better.
#
# The bracket is in the key because it is the axis that discriminates: the
# anchor is `_ANCHOR_MARGIN` knots above the bad band, and how much density
# that skips depends on the pad's step, which is the bracket's log span over
# the pad's knot count. A table pinned to one bracket cannot see it, and the
# first version of this test was pinned to [1e-4, 1e4]. The rows below the
# first block are the brackets where that cost up to 3378x. Adding a bracket
# means adding rows; the table is the case list.
_UNANCHORED = {
    (1e-4, 1e4, 1.0, 4): 1.6e-03,
    (1e-4, 1e4, 1.0, 5): 7.4e-05,
    (1e-4, 1e4, 1.0, 6): 2.6e-05,
    (1e-4, 1e4, 1.0, 7): 8.3e-06,
    (1e-4, 1e4, 1.0, 8): 8.9e-07,
    (1e-4, 1e4, 1.0, 16): 1.5e-07,
    (1e-4, 1e4, 1.0, 64): 1.5e-07,
    (1e-4, 1e4, 2.5, 4): 3.7e-06,
    (1e-4, 1e4, 2.5, 5): 1.5e-06,
    (1e-4, 1e4, 2.5, 6): 1.5e-06,
    (1e-4, 1e4, 2.5, 7): 1.3e-06,
    (1e-4, 1e4, 2.5, 8): 1.3e-06,
    (1e-4, 1e4, 2.5, 16): 1.3e-06,
    (1e-4, 1e4, 2.5, 64): 1.3e-06,
    (1e-4, 1e4, 2.7, 4): 6.9e-04,
    (1e-4, 1e4, 2.7, 5): 7.3e-04,
    (1e-4, 1e4, 2.7, 6): 6.4e-04,
    (1e-4, 1e4, 2.7, 7): 7.0e-04,
    (1e-4, 1e4, 2.7, 8): 7.2e-04,
    (1e-4, 1e4, 2.7, 16): 7.1e-04,
    (1e-4, 1e4, 2.7, 64): 6.8e-04,
    (1e-4, 1e4, 2.9, 4): 1.1e-01,
    (1e-4, 1e4, 2.9, 5): 1.1e-01,
    (1e-4, 1e4, 2.9, 6): 1.1e-01,
    (1e-4, 1e4, 2.9, 7): 1.2e-01,
    (1e-4, 1e4, 2.9, 8): 1.2e-01,
    (1e-4, 1e4, 2.9, 16): 1.2e-01,
    (1e-4, 1e4, 2.9, 64): 1.2e-01,
    (1e-5, 1e5, 2.5, 5): 3.8e-06,
    (1e-5, 1e5, 2.5, 6): 4.8e-06,
    (1e-5, 1e5, 2.5, 7): 4.6e-06,
    (1e-5, 1e5, 2.7, 5): 1.6e-03,
    (1e-5, 1e5, 2.7, 6): 1.5e-03,
    (1e-5, 1e5, 2.7, 7): 1.3e-03,
    (1e-6, 1e6, 2.5, 5): 1.3e-05,
    (1e-6, 1e6, 2.5, 6): 1.6e-05,
    (1e-6, 1e6, 2.5, 7): 1.3e-05,
    (1e-6, 1e6, 2.7, 5): 3.0e-03,
    (1e-6, 1e6, 2.7, 6): 2.9e-03,
    (1e-6, 1e6, 2.7, 7): 2.8e-03,
    (1e-4, 1e2, 2.5, 5): 8.8e-07,
    (1e-4, 1e2, 2.5, 6): 1.7e-06,
    (1e-4, 1e2, 2.5, 7): 1.3e-06,
    (1e-4, 1e2, 2.7, 5): 6.9e-04,
    (1e-4, 1e2, 2.7, 6): 6.5e-04,
    (1e-4, 1e2, 2.7, 7): 7.1e-04,
    (1e-3, 1e3, 2.5, 5): 4.1e-08,
    (1e-3, 1e3, 2.5, 6): 4.0e-08,
    (1e-3, 1e3, 2.5, 7): 2.8e-07,
    (1e-3, 1e3, 2.7, 5): 3.7e-04,
    (1e-3, 1e3, 2.7, 6): 3.3e-04,
    (1e-3, 1e3, 2.7, 7): 3.5e-04,
}


@pytest.mark.parametrize(("r_min", "r_max", "gamma", "n_r"), _UNANCHORED)
def test_anchoring_is_never_worse_than_an_unanchored_solve(
    r_min, r_max, gamma, n_r
) -> None:
    r"""Anchoring must not cost accuracy at any resolution, cusp or bracket.

    REGRESSION, three times over, each one a guard bounding the wrong thing.

    First the tail's activity gate was normalised by ``max|rho_col|`` over
    the whole column while testing ``rho_col[i0]``; before anchoring those
    were the same point for a cusp, so the gate could never fire. With an
    anchor it rejected once the ratio fell under ``sqrt(eps)``, dropping the
    tail -- the failure anchoring exists to prevent, re-created by its own
    margin.

    Then the fix for *that* exposed a bound on the slope fit's
    self-consistency, set first at 1.0 and then at 0.1 by measurement. Both
    bounded the fit's error *relative to the tail*, which does not reach the
    answer on its own: what reaches it is that error times how much larger
    the modelled band is than the band actually lost, and that factor is set
    by the grid, which the ratio never sees. 1.0 cost 99458x at
    ``gamma = 2.5, n_r = 5``; 0.1 cost 3378x at ``gamma = 2.5, n_r = 6`` over
    [1e-5, 1e5]. The gate is now the break-even test between the two, with no
    constant to tune.

    The sweep matters as much as the rule. Each defect lived where the
    previous version of this test did not look: at ``n_r`` 6 and 7 when it
    ran ``[4, 5, 8, 16, ...]``, at ``gamma = 2.5`` when it swept only 2.9,
    and at every bracket but one when it was pinned to [1e-4, 1e4]. A
    parameter sampled around its failure is not swept.

    The bound is `_UNANCHORED` with a 1.5x allowance for arithmetic
    reordering, floored at five float32 ulps -- not a flat constant, which
    would have let the 1.5e-06 -> 1.5e-01 case through at any threshold loose
    enough to pass ``gamma = 2.9`` at all. The allowance started at 3x and
    came down by measurement, and the floor exists because a ratio between
    two round-off numbers measures the platform rather than the solve; the
    comment at the assertion records both.
    """
    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)

    mass = 4.0 * jnp.pi / (3.0 - gamma)

    with jax.enable_x64(False):  # noqa: FBT003
        r_knots = jnp.geomspace(r_min, r_max, n_r)
        got = build_expansion(
            _dehnen_slope_in_t,
            r_knots,
            0,
            keys,
            n_theta,
            n_phi,
            jnp.asarray(gamma),
            jnp.asarray(1.0),
        )["phi_lm"][:, 0]
        assert jnp.all(jnp.isfinite(got)), got
        phi = jnp.asarray(got, dtype=float) / jnp.sqrt(4.0 * jnp.pi)

    r = jnp.asarray(r_knots, dtype=float)
    exact = -(mass / (2.0 - gamma)) * (1.0 - (r / (r + 1.0)) ** (2.0 - gamma))
    err = float(jnp.max(jnp.abs(phi - exact)) / jnp.max(jnp.abs(exact)))

    # 1.5x, not 3x. The worst row runs at 0.998 of its tabulated value, so 3x
    # was pure slack -- it let a gate loosened by a whole nat through at
    # (1e-4, 1e4, 2.5, 7), where the error doubles to 2.7e-06 against a 1.3e-06
    # row. 1.5x still leaves every row its measured headroom.
    #
    # Floored at a few float32 ulps, because a ratio between two round-off
    # numbers measures the platform, not the solve. Five of these rows sit
    # under it -- (1e-3, 1e3, 2.5, 5) is 4.1e-08, a third of an ulp -- and the
    # first CI run after the 3x allowance came down failed there on macOS at
    # 1.193e-07, which is 1.00 ulp exactly. The floor clears every row a
    # mutation has to beat, so it costs no sensitivity: the (2.5, 7) case
    # above stays bounded at 1.9e-06, well above it.
    bound = max(1.5 * _UNANCHORED[(r_min, r_max, gamma, n_r)], 5.0 * _F32_EPS)
    assert err < bound, (
        r_min,
        r_max,
        gamma,
        n_r,
        err,
    )


_CORE_GAMMA, _CORE_BETA, _CORE_ALPHA, _CORE_RS, _CORE_AMP = 2.95, 5.0, 1.0, 50.0, 1e17


def _cored_cusp_rc_in_t(xyz, t):
    """Evaluate a broken power law whose core radius is the *traced* ``t``."""
    r = safe_vector_norm(xyz)
    m = jnp.sqrt(r**2 + t**2)
    return (
        _CORE_AMP
        * m ** (-_CORE_GAMMA)
        * (1.0 + (m / _CORE_RS) ** _CORE_ALPHA)
        ** (-(_CORE_BETA - _CORE_GAMMA) / _CORE_ALPHA)
    )


@pytest.mark.parametrize(
    ("rc_over_r_overflow", "ceiling"), [(0.1, 2.0), (0.3, 3.0), (1.0, 3.0), (3.0, 1e-1)]
)
def test_a_core_at_the_overflow_radius_degrades_no_further(
    rc_over_r_overflow, ceiling
) -> None:
    r"""Pin the resonance where anchoring is *worse*, so it cannot grow.

    The anchoring gate reads the slope fit at the anchor and above it. The
    band the tail extrapolates across lies entirely *below* the anchor and
    holds no sample -- it is by construction the band that was zeroed. So a
    density that is a clean power law above the float32 overflow radius and
    something else below it passes the gate with ``|s1 - s0|`` at round-off:
    maximal confidence drawn from an absence of data.

    The worst such density puts its *core* at the overflow radius. Above it
    the profile is a pure cusp and the fit is perfect; below it the slope
    goes to zero and the extrapolated :math:`r^{-\gamma}` over-counts the
    inner mass without bound. An unanchored solve drops the band instead and
    under-counts, which is bounded by the band's true mass.

    The rows below sample where the density's curvature is still visible to
    the gate. They are not the whole problem: it is not a narrow resonance,
    and a core far under the overflow radius is worse in relative terms (see
    the family comment above `_FAMILY`). Measured float32 against float64 at
    :math:`\gamma = 2.95`, this branch against an unanchored build:

    ======================= ============ ============ =======
    ``rc`` / overflow radius  anchored     unanchored   ratio
    ======================= ============ ============ =======
    0.1                       1.1e-01      1.3e-02      8.4x
    0.3                       2.1e-01      1.3e-02      16x
    1.0                       1.4e+00      1.2e-02      119x
    3.0                       identical    identical    1.0
    ======================= ============ ============ =======

    At 3x and beyond nothing overflows, so nothing anchors and the two builds
    agree to the bit. At 1.0 a usable 1.2% answer becomes a 143% one.

    No gate on the retained samples can fix this, because the evidence is
    gone. This test exists so the regime it samples cannot deepen unnoticed,
    and so the 3.0 row cannot start anchoring.
    """
    gamma, amp, n_r = _CORE_GAMMA, _CORE_AMP, 8
    r_overflow = (float(jnp.finfo(jnp.float32).max) / amp) ** (-1.0 / gamma)
    rc = r_overflow * rc_over_r_overflow

    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)

    def build(r_knots):
        return build_expansion(
            _cored_cusp_rc_in_t,
            r_knots,
            0,
            keys,
            n_theta,
            n_phi,
            jnp.asarray(rc),
            jnp.asarray(1.0),
        )["phi_lm"][:, 0]

    ref = jnp.asarray(build(jnp.geomspace(1e-3, 1e2, n_r)), dtype=float)
    assert ref.dtype == jnp.float64, "the reference needs x64; see the pytest env"

    with jax.enable_x64(False):  # noqa: FBT003
        got = build(jnp.geomspace(1e-3, 1e2, n_r))
        assert jnp.all(jnp.isfinite(got)), got
        f32 = jnp.asarray(got, dtype=float)

    err = float(jnp.max(jnp.abs(f32 - ref)) / jnp.max(jnp.abs(ref)))
    assert err < ceiling, (rc_over_r_overflow, err)


# A randomly generated family of broken power laws, with their measured
# float32 error. The generator is `prop/build_sweep.py`'s `case()`: inner
# slope `gamma`, outer `beta`, break sharpness `alpha` and radius `rs`, an
# optional core `rc`, and an amplitude spanning 1e-6 to 1e30 -- the amplitude
# is what moves the float32 overflow radius, and therefore how much of the
# pad gets zeroed and where the tail has to be anchored.
#
# These are a net, not a theorem. Every other test here fixes the density and
# varies the grid; this one varies the density, because the one case where
# anchoring loses was found by hand-picking a shape (a core hidden under the
# overflow radius) rather than by any grid. Over 291 such densities run
# through `build_expansion` on this branch and on an unanchored build, three
# regressed past 1.5x -- worst 4.21x -- and all three were cored. 167 improved
# by more than 2x, best 1.4e4x.
#
# That is NOT the ceiling. The family does not sample cores or breaks far
# under the overflow radius, which is where anchoring loses most. A targeted
# sweep (float32, [1e-4, 1e4], n_r = 64, unit amplitude) measured, anchored
# against unanchored: a cored cusp at gamma = 2.99 with rc = 1e-13 is 2.7
# against 1.9e-2 (140x), 3.4 against 0.17 at rc = 1e-15 (20x), and still 2.6
# against 0.32 at rc = 1e-18 (8x); a broken power law (gamma 2.9 outside, 0.5
# inside, break at 1e-13) is 1.2e-1 against 2.1e-4 (570x). Below about
# rc = 1e-17 the float32 samples equal a pure cusp's, so no gate can tell the
# two apart; build in float64 if the density has structure there.
#
# Bounds are 2x the measured value, rounded up. A failure here means the
# family's behaviour moved, which is worth a look even when the new number is
# better -- retune the row deliberately rather than loosening the bound.
_FAMILY = {
    (2.773, 3.225, 2.729, 32.3, 5.299e-08, 2.389e19, 32, -2.662, 4.471, 0): 1.7e-01,
    (2.829, 5.846, 0.575, 95.09, 6.492e-12, 7.818e11, 7, -3.256, 1.927, 0): 2e-01,
    (2.155, 3.904, 1.498, 0.7139, 5.128e-10, 7.447e19, 8, -1.528, 2.970, 0): 1.4e-06,
    (2.848, 5.840, 1.432, 57.66, 1.326e-08, 1.731e21, 5, -3.759, 2.931, 0): 5.4e-01,
    (2.850, 4.953, 2.737, 0.1427, 1.445e-11, 1.238e13, 12, -5.051, 4.821, 0): 2.8e-01,
    (2.507, 5.046, 2.293, 3.3, 4.566e-09, 2.177e08, 32, -5.746, 2.065, 0): 7.8e-07,
    (2.780, 5.528, 0.781, 2.601, 3.48e-06, 5.419e17, 8, -1.193, 3.329, 0): 4.8e-07,
    (2.387, 5.178, 1.612, 0.3256, 2.327e-12, 2.486e24, 8, -4.062, 3.740, 0): 9e-02,
    (2.168, 3.834, 1.817, 0.1355, 8.201e-06, 2.011e13, 128, -2.046, 5.367, 0): 6.4e-07,
    (2.824, 3.065, 2.382, 17.46, 0, 1.204e09, 64, -5.929, 4.142, 4): 2.7e-05,
    (2.766, 3.847, 2.463, 31.08, 0, 5.845e08, 64, -5.971, 2.158, 0): 1.9e-06,
    (2.585, 5.926, 2.277, 81.07, 0, 5.973e12, 6, -5.582, 3.195, 0): 2.6e-06,
    (2.719, 5.065, 2.370, 0.06005, 0, 2.357e26, 64, -3.785, 4.880, 4): 2e-05,
    (2.836, 3.147, 2.796, 2.795, 0, 2.946e14, 128, -3.180, 4.686, 0): 2e-06,
    (2.017, 4.551, 0.718, 13.97, 0, 5.565e28, 64, -4.485, 1.639, 4): 3.3e-05,
    (2.778, 3.170, 1.288, 2.511, 0, 328, 128, -1.596, 4.805, 0): 1.6e-06,
    (2.545, 3.332, 2.821, 21.79, 0, 2.111e15, 8, -2.532, 2.534, 0): 1.5e-06,
    (2.789, 4.810, 1.324, 55.79, 0, 3.317e12, 7, -1.173, 3.877, 0): 1.7e-05,
    (1.665, 3.944, 1.196, 0.1046, 3.548e-07, 8.431e13, 12, -2.037, 4.111, 4): 6.5e-07,
    (2.963, 3.773, 0.901, 2.819, 0, 1.925e-05, 6, -3.669, 5.586, 0): 8.7e-01,
    (0.741, 5.905, 1.038, 4.865, 0, 2.929e25, 5, -5.342, 5.225, 2): 7.1e-07,
    (1.302, 5.266, 0.563, 0.3081, 0, 0.02655, 8, -2.711, 3.141, 4): 3.2e-07,
    (1.804, 5.625, 1.361, 2.297, 8.12e-09, 4.87e12, 12, -1.454, 1.755, 2): 3.2e-07,
    (2.475, 4.563, 2.315, 0.08048, 0, 1.182e07, 5, -4.270, 5.741, 0): 4.9e-05,
    (1.946, 4.309, 2.695, 0.4432, 5.7e-12, 3.019e09, 6, -1.245, 2.255, 0): 1.7e-07,
}


@partial(jax.jit, static_argnames=("l_max",))
def _family_build(params, r_knots, l_max):
    """`build_expansion` of a broken power law whose shape is a traced array.

    ``params`` is ``(-gamma, -(beta - gamma) / alpha, alpha, rs, rc**2, amp)``
    with the two derived entries formed in double precision by the caller, as
    the closure it replaces formed them. Built inside one jit, rows that share
    ``(n_r, l_max, dtype)`` share a compile instead of paying one each.
    """
    neg_gamma, outer, alpha, rs, rc2, amp = params
    keys = lm_keys(l_max, "spherical" if l_max == 0 else "zrotation_zreflection")
    n_theta, n_phi = default_angular_resolution(l_max)
    q = 1.0 if l_max == 0 else 0.8

    def rho(xyz, t):
        r = safe_vector_norm(
            jnp.stack([xyz[..., 0], xyz[..., 1], xyz[..., 2] / q], axis=-1)
        )
        m = jnp.sqrt(r**2 + rc2)
        return amp * m**neg_gamma * (1.0 + (m / rs) ** alpha) ** outer

    return build_expansion(
        rho, r_knots, l_max, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
    )["phi_lm"]


@pytest.mark.parametrize(
    ("gamma", "beta", "alpha", "rs", "rc", "amp", "n_r", "lo_e", "hi_e", "l_max"),
    _FAMILY,
)
def test_a_random_density_family_stays_within_its_measured_error(
    gamma, beta, alpha, rs, rc, amp, n_r, lo_e, hi_e, l_max
) -> None:
    """Float32 must not drift on a family of densities nobody hand-picked."""
    shape = [-gamma, -(beta - gamma) / alpha, alpha, rs, rc**2, amp]

    ref = jnp.asarray(
        _family_build(
            jnp.asarray(shape), jnp.geomspace(10.0**lo_e, 10.0**hi_e, n_r), l_max
        ),
        dtype=float,
    )
    assert ref.dtype == jnp.float64, "the reference needs x64; see the pytest env"
    assert jnp.all(jnp.isfinite(ref)), "float64 reference is not finite"

    with jax.enable_x64(False):  # noqa: FBT003
        got = _family_build(
            jnp.asarray(shape), jnp.geomspace(10.0**lo_e, 10.0**hi_e, n_r), l_max
        )
        assert got.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(got)), got
        f32 = jnp.asarray(got, dtype=float)

    err = float(jnp.max(jnp.abs(f32 - ref)) / jnp.max(jnp.abs(ref)))
    # Floored at five float32 ulps, as `_UNANCHORED` is: seven rows are
    # bounded below that, and a bound under a few ulps measures the platform.
    # `XLA_FLAGS=--xla_cpu_enable_fast_math=true` took one of them 2x over.
    bound = max(
        _FAMILY[(gamma, beta, alpha, rs, rc, amp, n_r, lo_e, hi_e, l_max)],
        5.0 * _F32_EPS,
    )
    assert err < bound, err

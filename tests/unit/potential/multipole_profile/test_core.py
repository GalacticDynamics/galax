"""Tests for the `MultipoleProfilePotential` class."""

import warnings

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.params as gpp
from galax.potential import CompositePotential, HernquistPotential, Symmetry
from galax.potential._src.base import default_constants
from galax.potential._src.builtin.multipole_profile.build import build_expansion
from galax.potential._src.builtin.multipole_profile.core import (
    AbstractMultipoleProfilePotential,
    MultipoleProfilePotential,
)
from galax.potential._src.builtin.multipole_profile.interp import _interpolate
from galax.potential._src.harmonic import default_angular_resolution, lm_keys
from galax.potential._src.utils import safe_vector_norm

XFAIL_T = (
    "batched `t` on a grid build: the expansion's hardcoded `axis=1` concat "
    "cannot take the time axis the interpolated coefficients gain -- "
    "https://github.com/GalacticDynamics/galax/issues/938"
)
"""Why `potential`/`density` are expected to fail on a batched `t`."""

G_GALACTIC = float(default_constants["G"].decompose(u.unitsystem("galactic")).value)
"""G in kpc^3 / (Msun Myr^2), from the same constant the potentials use.

Derived rather than written out: the literal it replaced had already drifted
from `default_constants` by 2.5e-8 relative, which is a systematic error
sitting inside the tolerance of every analytic comparison below.
"""

HERNQUIST = {"m_tot": u.Q(1e12, "Msun"), "r_s": u.Q(10.0, "kpc")}


def _reference() -> gp.HernquistPotential:
    return gp.HernquistPotential(**HERNQUIST, units="galactic")


def _hernquist_potential() -> MultipoleProfilePotential:
    """Expansion of a 1e12 Msun, 10 kpc Hernquist sphere, in galactic units."""
    m_tot, r_s = 1e12, 10.0

    def rho(xyz, t):
        r = jnp.linalg.norm(xyz, axis=-1)
        return m_tot / (2.0 * jnp.pi) * r_s / (r * (r + r_s) ** 3)

    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)
    r = jnp.geomspace(1e-2, 1e4, 256)
    coeffs = build_expansion(
        rho, r, 0, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(G_GALACTIC)
    )
    return MultipoleProfilePotential(
        r_knots=u.Q(r, "kpc"),
        phi_lm=u.Q(coeffs["phi_lm"], "kpc2 / Myr2"),
        dphi_lm=u.Q(coeffs["dphi_lm"], "kpc2 / Myr2"),
        d2phi_lm=u.Q(coeffs["d2phi_lm"], "kpc2 / Myr2"),
        phi_asympt_powers=u.Q(coeffs["phi_asympt_powers"], ""),
        phi_asympt_scales=u.Q(coeffs["phi_asympt_scales"], "kpc2 / Myr2"),
        rho_residual_lm=u.Q(coeffs["rho_residual_lm"], "Msun / kpc3"),
        drho_residual_lm=u.Q(coeffs["drho_residual_lm"], "Msun / kpc3"),
        rho_amplitude=u.Q(coeffs["rho_amplitude"], "Msun / kpc3"),
        rho_alpha=u.Q(coeffs["rho_alpha"], ""),
        l_max=0,
        symmetry="spherical",
        units="galactic",
    )


def test_is_an_abstract_multipole_profile_potential() -> None:
    pot = _hernquist_potential()
    assert isinstance(pot, AbstractMultipoleProfilePotential)
    assert isinstance(pot, gp.AbstractSinglePotential)


def test_exposes_expansion_configuration() -> None:
    pot = _hernquist_potential()
    assert pot.l_max == 0
    assert pot.lm_keys == ((0, 0),)
    assert pot.symmetry is Symmetry.SPHERICAL


def test_params_carries_every_coefficient() -> None:
    """`_params` is the contract between the class and the functions."""
    p = _hernquist_potential()._params(u.Q(0.0, "Gyr"))
    assert set(p) == {
        "r_knots",
        "phi_lm",
        "dphi_lm",
        "d2phi_lm",
        "phi_asympt_powers",
        "phi_asympt_scales",
        "rho_residual_lm",
        "drho_residual_lm",
        "rho_alpha",
        "rho_amplitude",
    }


def test_potential_matches_hernquist_with_units() -> None:
    pot, hern = _hernquist_potential(), _reference()
    x = u.Q([1.0, 2.0, 3.0], "kpc")
    got, exp = pot.potential(x, t=0), hern.potential(x, t=0)
    assert jnp.isclose(got, exp, rtol=1e-3, atol=u.Q(0.0, exp.unit))


def test_density_matches_hernquist_with_units() -> None:
    pot, hern = _hernquist_potential(), _reference()
    x = u.Q([1.0, 2.0, 3.0], "kpc")
    got, exp = pot.density(x, t=0), hern.density(x, t=0)
    assert jnp.isclose(got, exp, rtol=1e-3, atol=u.Q(0.0, exp.unit))


def test_gradient_matches_hernquist_with_units() -> None:
    """Forces come from `jax.grad`; no hand-coded derivative is needed."""
    pot, hern = _hernquist_potential(), _reference()
    x = u.Q([1.0, 2.0, 3.0], "kpc")
    got, exp = pot.gradient(x, t=0), hern.gradient(x, t=0)
    assert jnp.allclose(got, exp, rtol=5e-3, atol=u.Q(0.0, exp.unit))


def test_potential_is_time_independent_for_constant_coefficients() -> None:
    pot = _hernquist_potential()
    x = u.Q([1.0, 2.0, 3.0], "kpc")
    p0, p5 = pot.potential(x, t=u.Q(0.0, "Gyr")), pot.potential(x, t=u.Q(5.0, "Gyr"))
    assert jnp.isclose(p0, p5, atol=u.Q(0.0, p0.unit))


def test_from_potential_reproduces_hernquist() -> None:
    """The headline use: wrap an existing potential's density."""
    hern = _reference()
    pot = MultipoleProfilePotential.from_potential(
        hern,
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e4, "kpc"),
        n_r=256,
        l_max=0,
        symmetry="spherical",
    )
    x = u.Q([1.0, 2.0, 3.0], "kpc")
    got_pot, exp_pot = pot.potential(x, t=0), hern.potential(x, t=0)
    got_rho, exp_rho = pot.density(x, t=0), hern.density(x, t=0)
    assert jnp.isclose(got_pot, exp_pot, rtol=1e-3, atol=u.Q(0.0, exp_pot.unit))
    assert jnp.isclose(got_rho, exp_rho, rtol=1e-3, atol=u.Q(0.0, exp_rho.unit))


def test_from_potential_inherits_units_and_constants() -> None:
    hern = _reference()
    pot = MultipoleProfilePotential.from_potential(
        hern, r_min=u.Q(1e-2, "kpc"), r_max=u.Q(1e3, "kpc"), n_r=32, l_max=0
    )
    assert pot.units == hern.units
    assert pot.constants["G"] == hern.constants["G"]


def test_from_density_accepts_a_plain_callable() -> None:
    m_tot, r_s = 1e12, 10.0

    def rho(xyz, t):
        r = jnp.linalg.norm(xyz, axis=-1)
        return m_tot / (2.0 * jnp.pi) * r_s / (r * (r + r_s) ** 3)

    pot = MultipoleProfilePotential.from_density(
        rho,
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e4, "kpc"),
        n_r=256,
        l_max=0,
        symmetry="spherical",
        units="galactic",
    )
    x = u.Q([1.0, 2.0, 3.0], "kpc")
    got, exp = pot.potential(x, t=0), _reference().potential(x, t=0)
    assert jnp.isclose(got, exp, rtol=1e-3, atol=u.Q(0.0, exp.unit))


def test_from_density_accepts_a_bound_density_method() -> None:
    """`rho_fn` is a jit static argument, so it must not need to be hashable.

    An equinox bound method closes over array-valued parameters and is
    unhashable, which used to raise "Non-hashable static arguments are not
    supported" -- `from_potential` escaped only by wrapping in a lambda.
    """
    hern = _reference()
    pot = MultipoleProfilePotential.from_density(
        hern._density,
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e4, "kpc"),
        n_r=256,
        l_max=0,
        symmetry="spherical",
        units=hern.units,
    )
    x = u.Q([1.0, 2.0, 3.0], "kpc")
    got, exp = pot.potential(x, t=0), hern.potential(x, t=0)
    assert jnp.isclose(got, exp, rtol=1e-3, atol=u.Q(0.0, exp.unit))


def test_from_density_defaults_angular_resolution_from_l_max() -> None:
    def rho(xyz, t):
        return jnp.exp(-jnp.linalg.norm(xyz, axis=-1))

    pot = MultipoleProfilePotential.from_density(
        rho,
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e2, "kpc"),
        n_r=32,
        l_max=4,
        symmetry="plane_reflection",
        units="galactic",
    )
    assert pot.l_max == 4
    assert pot.symmetry is Symmetry.PLANE_REFLECTION
    assert len(pot.lm_keys) == 6


def test_from_density_rejects_an_unknown_symmetry() -> None:
    def rho(xyz, t):
        return jnp.exp(-jnp.linalg.norm(xyz, axis=-1))

    with pytest.raises(ValueError, match="Unknown symmetry"):
        MultipoleProfilePotential.from_density(
            rho,
            r_min=u.Q(1e-2, "kpc"),
            r_max=u.Q(1e2, "kpc"),
            l_max=2,
            symmetry="cubic",
            units="galactic",
        )


def test_from_potential_rejects_time_dependent_parameters() -> None:
    """A *single-time* expansion cannot track a time-varying source.

    Better to refuse than to hand back a potential silently inconsistent with
    its own density. The refusal is now conditional on the build being at a
    single time: `test_a_time_varying_source_builds_on_a_time_grid` covers the
    1-D ``t`` case, which is how such a source is tracked.
    """
    hern = gp.HernquistPotential(
        m_tot=gpp.LinearParameter(
            slope=u.Q(1e11, "Msun / Gyr"),
            point_time=u.Q(0.0, "Gyr"),
            point_value=u.Q(1e12, "Msun"),
        ),
        r_s=u.Q(10.0, "kpc"),
        units="galactic",
    )
    with pytest.raises(ValueError, match="vary with time"):
        MultipoleProfilePotential.from_potential(
            hern, r_min=u.Q(1e-2, "kpc"), r_max=u.Q(1e3, "kpc"), n_r=32, l_max=0
        )


def test_from_density_rejects_n_r_less_than_4() -> None:
    """`n_r` must be at least 4.

    The boundary power-law slopes are fitted over the innermost and outermost
    three knots (`poisson.py` slices ``[:3]`` and ``[-3:]``, `funcs.py`
    ``[:3]``), so fewer than four knots leaves the two windows identical and
    the fits meaningless.
    """

    def rho(xyz, t):
        return jnp.exp(-jnp.linalg.norm(xyz, axis=-1))

    with pytest.raises(ValueError, match="n_r must be >= 4"):
        MultipoleProfilePotential.from_density(
            rho,
            r_min=u.Q(1e-2, "kpc"),
            r_max=u.Q(1e2, "kpc"),
            n_r=2,
            l_max=0,
            units="galactic",
        )


@pytest.mark.parametrize("r_min_kpc", [0.0, -1.0])
def test_from_density_rejects_a_non_positive_r_min(r_min_kpc) -> None:
    """The radial grid is log-spaced, so the bracket must be positive.

    Without this guard, `log(r_min)` produces -inf or NaN and the failure
    surfaces much later as an opaque `JaxRuntimeError` from inside the jitted
    build, with no indication of which argument was at fault.
    """

    def rho(xyz, t):
        return jnp.exp(-jnp.linalg.norm(xyz, axis=-1))

    with pytest.raises(ValueError, match="r_min must be > 0"):
        MultipoleProfilePotential.from_density(
            rho,
            r_min=u.Q(r_min_kpc, "kpc"),
            r_max=u.Q(1e2, "kpc"),
            n_r=32,
            l_max=0,
            units="galactic",
        )


def test_from_density_rejects_r_min_greater_or_equal_to_r_max() -> None:
    """r_min must be strictly less than r_max."""

    def rho(xyz, t):
        return jnp.exp(-jnp.linalg.norm(xyz, axis=-1))

    with pytest.raises(ValueError, match="r_min must be < r_max"):
        MultipoleProfilePotential.from_density(
            rho,
            r_min=u.Q(1e2, "kpc"),
            r_max=u.Q(1e2, "kpc"),
            n_r=32,
            l_max=0,
            units="galactic",
        )


def _raw_fields(pot) -> dict:
    """Decompose a built potential into the raw arguments `__init__` takes."""
    t0 = u.Q(0.0, "Gyr")
    return {
        "r_knots": pot.r_knots(t0),
        "phi_lm": pot.phi_lm(t0),
        "dphi_lm": pot.dphi_lm(t0),
        "d2phi_lm": pot.d2phi_lm(t0),
        "phi_asympt_powers": pot.phi_asympt_powers(t0),
        "phi_asympt_scales": pot.phi_asympt_scales(t0),
        "rho_residual_lm": pot.rho_residual_lm(t0),
        "drho_residual_lm": pot.drho_residual_lm(t0),
        "rho_amplitude": pot.rho_amplitude(t0),
        "rho_alpha": pot.rho_alpha(t0),
        "l_max": pot.l_max,
        "symmetry": pot.symmetry,
        "units": pot.units,
    }


@pytest.mark.parametrize(
    ("label", "mutate", "match"),
    [
        (
            "truncated phi_lm",
            lambda f: {"phi_lm": f["phi_lm"][:-1]},
            "phi_lm must have shape",
        ),
        (
            "short rho_alpha",
            lambda f: {"rho_alpha": f["rho_alpha"][:-1]},
            "rho_alpha must have shape",
        ),
        (
            "wrong-shaped tail coefficients",
            lambda f: {"phi_asympt_powers": f["phi_asympt_powers"][:, :1]},
            "phi_asympt_powers must have shape",
        ),
        (
            "too few knots",
            lambda f: {"r_knots": f["r_knots"][:2]},
            "at least 4 entries",
        ),
        (
            "negative knot",
            lambda f: {"r_knots": f["r_knots"].at[0].set(u.Q(-1.0, "kpc"))},
            "strictly positive",
        ),
        (
            "unsorted knots",
            lambda f: {"r_knots": f["r_knots"][::-1]},
            "strictly increasing",
        ),
    ],
)
def test_check_init_rejects_inconsistent_coefficients(label, mutate, match) -> None:
    """`__init__` is public, so its invariants must be enforced there.

    Coefficients computed elsewhere can be loaded without a rebuild. Without
    these checks an inconsistent instance is accepted and fails much later
    inside `searchsorted`, `log` or a broadcast, far from the cause.
    """
    good = _hernquist_potential()
    fields = _raw_fields(good)
    assert isinstance(MultipoleProfilePotential(**fields), MultipoleProfilePotential)

    with pytest.raises(ValueError, match=match):
        MultipoleProfilePotential(**{**fields, **mutate(fields)})


def _flattened_density(xyz, t):
    """Return a triaxial exponential: angular structure in theta and in phi."""
    del t
    m = jnp.sqrt(xyz[..., 0] ** 2 + (xyz[..., 1] / 0.6) ** 2 + (xyz[..., 2] / 0.4) ** 2)
    return 1e10 * jnp.exp(-m)


def _time_scaled_density(xyz, t):
    """Return a density whose amplitude genuinely depends on ``t``."""
    r = jnp.linalg.norm(xyz, axis=-1)
    return 1e10 * (1.0 + t) * jnp.exp(-r)


def _max_rel(got, expect) -> float:
    return float(jnp.max(jnp.abs(got - expect)) / jnp.max(jnp.abs(expect)))


def test_from_density_uses_n_theta_and_n_phi_as_given() -> None:
    """The overrides must reach the quadrature, in the right slots.

    Compared against a direct `build_expansion` at the same resolution, and
    against one with the two deliberately transposed: an override that is
    ignored, or a pair that is silently swapped, fails one of the two. The
    transposed build differs by 1.2e-2 here, well clear of the 1e-3 bound.
    """
    n_theta, n_phi = 5, 15  # distinct, and neither is the l_max=2 default
    l_max, n_r = 2, 32
    keys = lm_keys(l_max, None)
    r = jnp.geomspace(1e-2, 1e2, n_r)
    args = (_flattened_density, r, l_max, keys)
    # `from_density`'s own G, so the comparison isolates the quadrature.
    g = jnp.asarray(default_constants["G"].decompose(u.unitsystem("galactic")).value)
    tail = (jnp.asarray(0.0), g)
    direct = build_expansion(*args, n_theta, n_phi, *tail)
    transposed = build_expansion(*args, n_phi, n_theta, *tail)

    pot = MultipoleProfilePotential.from_density(
        _flattened_density,
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e2, "kpc"),
        n_r=n_r,
        l_max=l_max,
        n_theta=n_theta,
        n_phi=n_phi,
        units="galactic",
    )
    got = u.ustrip("kpc2 / Myr2", pot.phi_lm(u.Q(0.0, "Gyr")))

    assert _max_rel(got, direct["phi_lm"]) < 1e-12
    assert _max_rel(got, transposed["phi_lm"]) > 1e-3


def test_from_density_builds_at_the_requested_time() -> None:
    """``t`` must reach the density, not be silently replaced by 0 Gyr."""
    kw = {
        "r_min": u.Q(1e-2, "kpc"),
        "r_max": u.Q(1e2, "kpc"),
        "n_r": 32,
        "l_max": 0,
        "symmetry": "spherical",
        "units": "galactic",
    }
    at_0 = MultipoleProfilePotential.from_density(_time_scaled_density, **kw)
    at_1 = MultipoleProfilePotential.from_density(
        _time_scaled_density, t=u.Q(1.0, "Gyr"), **kw
    )
    t0 = u.Q(0.0, "Gyr")
    phi_0 = u.ustrip("kpc2 / Myr2", at_0.phi_lm(t0))
    phi_1 = u.ustrip("kpc2 / Myr2", at_1.phi_lm(t0))

    # The density is linear in t, and galactic time is Myr: 1 Gyr -> 1 + 1000.
    assert jnp.allclose(phi_1, 1001.0 * phi_0, rtol=1e-10)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"l_max": -1}, "l_max must be >= 0"),
        ({"l_max": 2, "n_theta": 0}, "n_theta must be >= 1"),
        ({"l_max": 2, "n_phi": -3}, "n_phi must be >= 1"),
    ],
)
def test_from_density_rejects_a_nonsensical_resolution(kwargs, match) -> None:
    """Without these, `l_max=-1` selects no modes and fails much later.

    ``lm_keys(-1)`` is empty and numpy eventually raises "deg must be a
    positive integer" from inside the jitted build, with no hint as to which
    argument was at fault; ``n_theta=0`` is the same story.
    """

    def rho(xyz, t):
        return jnp.exp(-jnp.linalg.norm(xyz, axis=-1))

    with pytest.raises(ValueError, match=match):
        MultipoleProfilePotential.from_density(
            rho,
            r_min=u.Q(1e-2, "kpc"),
            r_max=u.Q(1e2, "kpc"),
            n_r=32,
            units="galactic",
            **kwargs,
        )


@pytest.mark.parametrize(
    ("r_min_kpc", "r_max_kpc"),
    [
        (float("nan"), 1e2),
        (1e-2, float("nan")),
        (float("inf"), 1e2),
        (1e-2, float("inf")),
    ],
)
def test_from_density_rejects_a_non_finite_bracket(r_min_kpc, r_max_kpc) -> None:
    """A non-finite bracket slips past both ordering guards.

    Every comparison against a ``nan`` is `False`, so ``nan <= 0.0`` and
    ``r_min >= nan`` both pass, and ``geomspace`` then returns an all-``nan``
    grid -- the opaque, far-from-the-cause failure the other two guards exist
    to prevent.
    """

    def rho(xyz, t):
        return jnp.exp(-jnp.linalg.norm(xyz, axis=-1))

    with pytest.raises(ValueError, match="must be finite"):
        MultipoleProfilePotential.from_density(
            rho,
            r_min=u.Q(r_min_kpc, "kpc"),
            r_max=u.Q(r_max_kpc, "kpc"),
            n_r=32,
            l_max=0,
            units="galactic",
        )


def test_from_density_rejects_a_non_scalar_bracket() -> None:
    """A non-scalar bracket must name the argument, not leak a `TypeError`.

    ``r_min`` / ``r_max`` are build-time configuration and go through
    ``float(...)``, which reports only "Only scalar arrays can be converted
    to Python scalars" -- true, but it does not say which of the two, nor
    that they were meant to be scalars in the first place.
    """

    def rho(xyz, t):
        return jnp.exp(-jnp.linalg.norm(xyz, axis=-1))

    with pytest.raises(ValueError, match="must be scalars"):
        MultipoleProfilePotential.from_density(
            rho,
            r_min=u.Q(jnp.asarray([1e-2, 2e-2]), "kpc"),
            r_max=u.Q(1e2, "kpc"),
            n_r=32,
            l_max=0,
            units="galactic",
        )


def test_an_array_time_builds_a_grid_and_does_not_average() -> None:
    """An array ``t`` must build one expansion per time, not average them.

    REGRESSION: `harmonic_coeffs` broadcasts ``t`` against the angular grid,
    so a non-scalar time never raised on its own -- it returned a
    time-*averaged* expansion presented as a single-time potential. For a
    density whose amplitude doubles between ``t = 0`` and ``t = 1 Gyr``,
    ``t = [0, 1]`` gave exactly the mean of the two builds: wrong values,
    silently.

    An array ``t`` now builds a grid, so the failure mode this guards is no
    longer "it raises" but "it returns the mean". Both endpoints are checked
    against their own single-time builds, and that is the whole test: the
    midpoint is deliberately *not* asserted, because for a linear-in-``t``
    amplitude the mean of the two builds is the correct answer there, so a
    midpoint check passes equally for a grid and for the averaging bug. Only
    the endpoints tell them apart.
    """

    def rho(xyz, t):
        return (1.0 + t) * jnp.exp(-jnp.linalg.norm(xyz, axis=-1))

    kw = {
        "r_min": u.Q(1e-2, "kpc"),
        "r_max": u.Q(1e2, "kpc"),
        "n_r": 32,
        "l_max": 0,
        "symmetry": "spherical",
        "units": "galactic",
    }
    ts = u.Q(jnp.asarray([0.0, 1000.0]), "Myr")
    grid = MultipoleProfilePotential.from_density(rho, t=ts, **kw)
    assert grid.phi_lm.func is _interpolate  # tabulated, not constant

    xyz = u.Q(jnp.asarray([2.0, 1.0, 0.5]), "kpc")

    def phi(pot, tv):
        return float(u.ustrip(u.unit("kpc2/Myr2"), pot.potential(xyz, u.Q(tv, "Myr"))))

    ends = [
        phi(MultipoleProfilePotential.from_density(rho, t=u.Q(tv, "Myr"), **kw), tv)
        for tv in (0.0, 1000.0)
    ]
    # Each endpoint is its own build, not the mean of the two.
    assert phi(grid, 0.0) == pytest.approx(ends[0], rel=1e-10)
    assert phi(grid, 1000.0) == pytest.approx(ends[1], rel=1e-10)
    # ...and they are far enough apart that averaging would be visible.
    assert abs(ends[1] - ends[0]) / abs(ends[0]) > 0.5


def test_from_density_rejects_a_multidimensional_time() -> None:
    """A 2-D ``t`` still has no meaning and must not reach the build.

    The silent-averaging hazard above is why this cannot simply be passed
    through to `harmonic_coeffs`.
    """

    def rho(xyz, t):
        return jnp.exp(-jnp.linalg.norm(xyz, axis=-1))

    with pytest.raises(ValueError, match="must be a scalar or 1-D"):
        MultipoleProfilePotential.from_density(
            rho,
            r_min=u.Q(1e-2, "kpc"),
            r_max=u.Q(1e2, "kpc"),
            n_r=16,
            l_max=0,
            units="galactic",
            t=u.Q(jnp.zeros((2, 2)), "Gyr"),
        )


@pytest.mark.parametrize("scale", [1.0, 1e-8], ids=["amp>1", "amp<1"])
def test_density_is_finite_in_float32_for_a_steep_cusp(scale) -> None:
    """The cusp background must not overflow at reachable radii.

    REGRESSION: the background is ``amplitude * exp(alpha * log(r/r0))`` and
    was unclamped. `galax` does not enable x64 on import, so float32 is the
    default, where ``exp`` overflows at ~88 -- and an ``r^-2`` cusp reaches
    that at the origin, giving ``inf``. Clamping the *exponent* alone is not
    enough: ``safe_vector_norm`` floors r at ``sqrt(tiny)``, so the exponent
    is large but finite there and it is the product with ``amplitude`` that
    overflows. Nor is bounding the exponent by ``ln_huge - log|amplitude|``:
    that is *looser* than ``ln_huge`` whenever ``|amplitude| < 1``, so
    ``exp`` overflows on its own -- hence the ``amp<1`` case, which the
    first version of this fix returned ``inf`` for.

    The density outside the knots is not continued and is documented as
    meaningless, so saturating costs nothing -- but an ``inf`` propagates,
    and one bad radius poisons a whole vmapped batch.
    """

    def rho(xyz, t):
        r = jnp.sqrt(jnp.sum(xyz**2, axis=-1))
        return scale / r**2.8 / (1.0 + r) ** 2

    pot = MultipoleProfilePotential.from_density(
        rho,
        r_min=u.Q(0.05, "kpc"),
        r_max=u.Q(20.0, "kpc"),
        n_r=64,
        l_max=0,
        symmetry="spherical",
        units="galactic",
    )

    r = jnp.asarray([0.0, 1e-30, 1e-10, 1.0, 1e9, 1e15], dtype=jnp.float32)
    x = u.Q(jnp.stack([r, jnp.zeros_like(r), jnp.zeros_like(r)], axis=-1), "kpc")
    got = pot.density(x, t=u.Q(0.0, "Gyr")).ustrip("Msun/kpc3")

    assert jnp.all(jnp.isfinite(got)), f"non-finite density: {got}"


@pytest.mark.parametrize(
    ("r_min", "r_max", "expect_warning"),
    [(0.05, 20.0, False), (1e-3, 1e3, False), (1e-6, 1e6, True)],
)
def test_from_density_warns_when_the_padding_is_capped(
    r_min, r_max, expect_warning
) -> None:
    """A capped pad is a silent accuracy loss, so say so.

    `build_expansion` pads to push the boundary power-law model out of the
    answer, and caps that reach at what the dtype can exponentiate. Past the
    cap the build still succeeds -- it just keeps more of the tail model than
    it meant to, and nothing in the result says so.

    Only this constructor can warn: `build_expansion` takes `r_knots` as a
    traced array, so the span is not known where the padding happens, while
    here `r_min` and `r_max` are concrete.

    Runs in float32 because that is where the cap binds -- `galax`'s default,
    and never the suite's. In float64 the budget is 699 e-folds, which the
    last case checks stays quiet.
    """
    hern = gp.HernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(10.0, "kpc"), units="galactic"
    )

    def build(x64):
        with jax.enable_x64(x64):
            return gp.MultipoleProfilePotential.from_density(
                hern.density,
                r_min=u.Q(r_min, "kpc"),
                r_max=u.Q(r_max, "kpc"),
                n_r=32,
                l_max=0,
                symmetry="spherical",
                units="galactic",
            )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build(False)  # noqa: FBT003
    got = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert bool(got) is expect_warning, [str(w.message) for w in got]

    # The same bracket in float64 has budget to spare and must stay quiet.
    with warnings.catch_warnings(record=True) as caught64:
        warnings.simplefilter("always")
        build(True)  # noqa: FBT003
    assert not [w for w in caught64 if issubclass(w.category, RuntimeWarning)]


def _growing_hernquist(xyz, t):
    """Hernquist of fixed mass whose scale radius grows linearly in time.

    ``rho0 = 1e12 / rs**3`` keeps ``M = 2 pi rho0 rs**3`` constant, so the
    closed form ``Phi = -G M / (r + rs)`` is exact at every time and the only
    thing varying is the shape. ``t`` arrives in the unit system's time unit
    (Myr for "galactic"), not in Gyr.
    """
    r = safe_vector_norm(xyz)
    rs = 1.0 + 0.01 * t
    return 1e12 / ((r / rs) * (1.0 + r / rs) ** 3) / rs**3


def _hernquist_phi(r, rs):
    """``-G M / (r + rs)`` for the density above."""
    return -G_GALACTIC * 2.0 * jnp.pi * 1e12 / (r + rs)


def test_time_grid_build_matches_the_closed_form_at_each_knot() -> None:
    """A time-grid expansion must reproduce the analytic potential in time.

    The density is a Hernquist of *constant mass* whose scale radius grows,
    so `_hernquist_phi` is exact at every time and this pins the whole
    time-grid path -- the `vmap` over `t`, the per-coefficient
    `time_interpolated_parameter`, and the interpolation -- against physics
    rather than against another `galax` build.
    """
    ts = u.Q(jnp.linspace(0.0, 400.0, 5), "Myr")
    pot = gp.MultipoleProfilePotential.from_density(
        _growing_hernquist,
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e4, "kpc"),
        n_r=256,
        l_max=0,
        symmetry="spherical",
        t=ts,
        units="galactic",
    )
    assert pot.phi_lm.func is _interpolate  # tabulated, not constant

    xyz = u.Q(jnp.asarray([3.0, 1.0, 2.0]), "kpc")
    r = jnp.linalg.vector_norm(u.ustrip(u.unit("kpc"), xyz))

    for tv in (0.0, 100.0, 200.0, 400.0):
        got = u.ustrip(u.unit("kpc2/Myr2"), pot.potential(xyz, u.Q(tv, "Myr")))
        want = _hernquist_phi(r, 1.0 + 0.01 * tv)
        assert abs(float(got - want) / float(want)) < 1e-4, (tv, got, want)


def test_time_grid_interpolates_between_knots() -> None:
    """Between knots the answer must be close to the closed form, not merely finite.

    A knot-only check would pass even if the interpolation returned the
    nearest knot, so this samples deliberately off-knot.
    """
    ts = u.Q(jnp.linspace(0.0, 400.0, 9), "Myr")
    pot = gp.MultipoleProfilePotential.from_density(
        _growing_hernquist,
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e4, "kpc"),
        n_r=256,
        l_max=0,
        symmetry="spherical",
        t=ts,
        units="galactic",
    )
    xyz = u.Q(jnp.asarray([3.0, 1.0, 2.0]), "kpc")
    r = jnp.linalg.vector_norm(u.ustrip(u.unit("kpc"), xyz))

    for tv in (37.0, 162.5, 333.3):  # none of these is a knot
        got = u.ustrip(u.unit("kpc2/Myr2"), pot.potential(xyz, u.Q(tv, "Myr")))
        want = _hernquist_phi(r, 1.0 + 0.01 * tv)
        assert abs(float(got - want) / float(want)) < 1e-3, (tv, got, want)


def test_time_grid_clamps_outside_the_grid() -> None:
    """Outside ``[t_0, t_-1]`` the expansion saturates; it does not extrapolate.

    Continuing the edge cubic in time would invent structure the tabulation
    knows nothing about, and for a coefficient that diverges fast. See
    `time_interpolated_parameter`.
    """
    ts = u.Q(jnp.linspace(0.0, 400.0, 5), "Myr")
    pot = gp.MultipoleProfilePotential.from_density(
        _growing_hernquist,
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e4, "kpc"),
        n_r=64,
        l_max=0,
        symmetry="spherical",
        t=ts,
        units="galactic",
    )
    xyz = u.Q(jnp.asarray([3.0, 1.0, 2.0]), "kpc")

    def phi(tv):
        return u.ustrip(u.unit("kpc2/Myr2"), pot.potential(xyz, u.Q(tv, "Myr")))

    assert phi(-5000.0) == pytest.approx(float(phi(0.0)), rel=1e-12)
    assert phi(9999.0) == pytest.approx(float(phi(400.0)), rel=1e-12)


@pytest.mark.parametrize(
    ("t", "match"),
    [
        (jnp.zeros((2, 2)), "must be a scalar or 1-D"),
        (jnp.asarray([1.0]), "at least 2 entries"),
        (jnp.asarray([0.0, 200.0, 100.0, 300.0]), "strictly increasing"),
        (jnp.asarray([0.0, 100.0, 100.0, 300.0]), "strictly increasing"),
        # `inf` satisfies `diff > 0` -- `inf - 100` is `inf` -- so finiteness
        # is a separate condition, and without it this builds and then makes
        # every interpolated value `nan`. `nan` needs no separate case: no
        # comparison involving it is true, so monotonicity already rejects it.
        (jnp.asarray([0.0, 100.0, jnp.inf]), "strictly increasing and finite"),
        (jnp.asarray([-jnp.inf, 100.0, 300.0]), "strictly increasing and finite"),
        (jnp.asarray([0.0, jnp.nan, 300.0]), "strictly increasing and finite"),
    ],
)
def test_time_grid_rejects_a_malformed_time(t, match: str) -> None:
    """A malformed time grid must be refused rather than quietly misused.

    All of these have to be caught here, because none of them fails on its
    own. `harmonic_coeffs` broadcasts ``t`` against the angular grid, so a
    2-D one would silently *average* the expansion over those times. And an
    out-of-order grid builds perfectly well, then interpolates against a
    `jnp.searchsorted` that assumes sortedness -- it brackets the query with
    some other interval and returns wrong values with no complaint.
    `interpax` even sorts internally when fitting the knot derivatives, so
    those look right too.

    Verified the hazard is real before guarding it: on a scrambled grid with
    arbitrary values, `eval_log_spline` disagrees with the sorted answer in
    every query tried. A smooth test function hides it -- the first attempt
    used ``1/(1 + 0.01 t)`` and the two index paths happened to land on
    numerically coincident cubics, agreeing to the last digit.
    """
    with pytest.raises(ValueError, match=match):
        gp.MultipoleProfilePotential.from_density(
            _growing_hernquist,
            r_min=u.Q(1e-2, "kpc"),
            r_max=u.Q(1e3, "kpc"),
            n_r=16,
            l_max=0,
            symmetry="spherical",
            t=u.Q(t, "Myr"),
            units="galactic",
        )


def test_a_time_varying_source_builds_on_a_time_grid() -> None:
    """A source whose own parameters vary is what the time grid is *for*.

    `_check_time_independent` rejects such a source for a single-time build,
    because one expansion cannot track it and returning one anyway would give
    a potential inconsistent with its own density. On a grid it is tracked,
    so the check is conditional and this is the case it allows through.
    """
    lp = gpp.LinearParameter(
        slope=u.Q(-1e8, "Msun/Myr"),
        point_time=u.Q(0, "Myr"),
        point_value=u.Q(1e12, "Msun"),
    )
    src = gp.HernquistPotential(m_tot=lp, r_s=u.Q(10.0, "kpc"), units="galactic")
    kw = {
        "r_min": u.Q(1e-2, "kpc"),
        "r_max": u.Q(1e3, "kpc"),
        "n_r": 64,
        "l_max": 0,
        "symmetry": "spherical",
    }

    with pytest.raises(ValueError, match="vary with time"):
        gp.MultipoleProfilePotential.from_potential(src, t=u.Q(0.0, "Myr"), **kw)

    pot = gp.MultipoleProfilePotential.from_potential(
        src, t=u.Q(jnp.linspace(0.0, 1000.0, 5), "Myr"), **kw
    )
    xyz = u.Q(jnp.asarray([5.0, 0.0, 0.0]), "kpc")
    for tv in (0.0, 250.0, 500.0, 1000.0):
        got = u.ustrip(u.unit("kpc2/Myr2"), pot.potential(xyz, u.Q(tv, "Myr")))
        want = u.ustrip(u.unit("kpc2/Myr2"), src.potential(xyz, u.Q(tv, "Myr")))
        assert abs(float(got - want) / float(want)) < 1e-6, (tv, got, want)


@pytest.mark.parametrize("l_max", [0, 4, 8])
@pytest.mark.parametrize("symmetry", ["none", "plane_reflection"])
def test_gradient_matches_autodiff(l_max: int, symmetry: str) -> None:
    """The analytic gradient and `jax.grad` must not drift apart.

    `MultipoleProfileMixin._gradient` overrides `AbstractPotential`'s
    `jax.grad` default with the chain rule written out (`expansion_gradient`),
    to avoid pushing an ``(n_modes,)`` cotangent back through the spline's
    gather. Both are `jax`'s own derivatives of the same code, so this pins
    them together permanently rather than checking the override once.

    The z-axis is included deliberately: the harmonics are evaluated from the
    Cartesian direction precisely so the derivative exists there, and a
    ``(theta, phi)`` form would give ``0/0`` for every ``m >= 1``.

    The source must be **aspherical**. An earlier version of this test built
    the expansion from a Hernquist sphere, whose projection is pure monopole,
    so every ``Phi_lm`` with ``l > 0`` was machine-zero and the whole
    angular term -- the `jax.vjp` and the tangential projection -- was
    multiplied by nothing. Deleting that term outright still passed all six
    cases. A triaxial source gives the higher-``l`` coefficients real
    amplitude, and the same mutation then fails.
    """
    src = gp.TriaxialNFWPotential(
        m=u.Q(1e12, "Msun"),
        r_s=u.Q(10.0, "kpc"),
        q1=1.0,
        q2=0.8,
        units="galactic",
    )
    pot = gp.MultipoleProfilePotential.from_potential(
        src,
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e3, "kpc"),
        n_r=128,
        l_max=l_max,
        symmetry=symmetry,
    )
    t = u.Q(0.0, "Gyr")

    xyz = jnp.concat(
        [
            jax.random.normal(jax.random.key(0), (16, 3)) * 8.0,
            # on-axis, and both signs of z
            jnp.asarray([[0.0, 0.0, 4.0], [0.0, 0.0, -2.5]]),
        ]
    )

    got = u.ustrip(u.unit("kpc/Myr2"), pot.gradient(u.Q(xyz, "kpc"), t))

    zero = jnp.asarray(0.0)
    ref = jax.vmap(jax.grad(lambda a: pot._potential(a, zero)))(xyz)

    assert jnp.all(jnp.isfinite(got)), got
    scale = jnp.max(jnp.abs(ref))
    assert float(jnp.max(jnp.abs(got - ref)) / scale) < 1e-11


def test_refining_the_time_grid_converges_on_a_direct_build() -> None:
    """The interpolation must converge, not merely be finite.

    Every other test here checks the grid build *at* its knots, where the
    interpolant reproduces the stored coefficients by construction and the
    interpolation is doing no work. This one checks between them, against the
    answer a direct single-time build gives, and asserts the error *falls*
    as the grid refines.

    That is the property that distinguishes interpolating from any cheaper
    thing that happens to be close: returning the nearest knot, or the mean
    of the bracketing pair, is finite and plausible at every resolution and
    converges at the wrong rate or not at all.

    The source varies in *shape*, not just amplitude -- the break radius
    moves -- so a build at one time cannot stand in for another. Measured
    maximum relative error over three field points at t = 0.61 of the span:
    9.8e-02, 3.3e-03, 4.2e-05 at n_t = 5, 9, 17, so each refinement gains
    well over an order. The bound below is 5x per refinement, which is loose
    against the ~30x measured and against the 16x a cubic predicts.
    """
    xyz = u.Q(jnp.asarray([[2.0, 1.0, 0.5], [0.3, -0.2, 0.9]]), "kpc")
    t0, t1 = 0.0, 1000.0
    omega = 2.0 * jnp.pi / 1000.0

    def rho(q, t):
        r = safe_vector_norm(q)
        scale = 1.0 + 0.4 * jnp.sin(omega * t)
        return 1.0 / (2.0 * jnp.pi) / r / (1.0 + r / scale) ** 3

    def build(t):
        return MultipoleProfilePotential.from_density(
            rho,
            r_min=u.Q(1e-2, "kpc"),
            r_max=u.Q(1e2, "kpc"),
            n_r=96,
            l_max=0,
            symmetry="spherical",
            units="galactic",
            t=t,
        )

    t_probe = t0 + 0.61 * (t1 - t0)
    want = u.ustrip(
        u.unit("kpc2/Myr2"),
        build(u.Q(t_probe, "Myr")).potential(xyz, u.Q(t_probe, "Myr")),
    )

    errs = []
    for n_t in (5, 9, 17):
        grid = build(u.Q(jnp.linspace(t0, t1, n_t), "Myr"))
        got = u.ustrip(u.unit("kpc2/Myr2"), grid.potential(xyz, u.Q(t_probe, "Myr")))
        errs.append(float(jnp.max(jnp.abs(got - want)) / jnp.max(jnp.abs(want))))

    assert errs[1] < errs[0] / 5.0, errs
    assert errs[2] < errs[1] / 5.0, errs


@pytest.mark.parametrize(
    "method",
    [
        "gradient",
        pytest.param("potential", marks=pytest.mark.xfail(strict=True, reason=XFAIL_T)),
        pytest.param("density", marks=pytest.mark.xfail(strict=True, reason=XFAIL_T)),
    ],
)
def test_a_time_grid_build_evaluates_at_a_batch_of_times(method: str) -> None:
    """Batched ``t`` works for `gradient` and is a known gap for the others.

    `_gradient` carries `vectorize_method`, so its body always sees a scalar
    ``t`` and a grid build evaluates fine. `_potential` and `_density` do
    not: a batched ``t`` reaches `_params`, whose coefficients are
    time-interpolated on a grid and so gain a leading time axis, which the
    expansion's positional ``axis=1`` concat and its broadcasts are not
    written for.

    It matters because a batch of times is how energies are taken along an
    orbit, so `potential_energy(pot, orbit)` does not yet work on a
    grid-built potential. `jax.vmap` over ``t`` is the workaround.

    Tracked in https://github.com/GalacticDynamics/galax/issues/938, which
    also records the direction and the dead end.

    Giving the two methods the decorator was tried and reverted: it fixes
    this, but `vectorize_method` broadcasts the *loop* dims of ``xyz`` and
    ``t``, which no other galax potential does, so a **constant**-parameter
    multipole started reporting shapes no other potential reports -- and
    `AbstractCompositePotential._potential` stacks its components rather
    than broadcasting them, so any composite containing one raised
    ``TypeError: Cannot concatenate arrays with different numbers of
    dimensions``. Breaking every composite to extend a new feature is the
    wrong trade; the fix belongs in the expansion's hardcoded axes, not in
    the evaluation path's shape contract.

    ``strict=True`` so that fixing it is noticed here rather than silently
    widening the contract.
    """

    def rho(xyz, t):
        r = safe_vector_norm(xyz)
        amp = 1.0 + 0.3 * jnp.sin(2.0 * jnp.pi * t / 400.0)
        return amp / (2.0 * jnp.pi) / r / (1.0 + r) ** 3

    pot = MultipoleProfilePotential.from_density(
        rho,
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e2, "kpc"),
        n_r=64,
        l_max=0,
        symmetry="spherical",
        units="galactic",
        t=u.Q(jnp.linspace(0.0, 400.0, 5), "Myr"),
    )

    xyz = u.Q(jnp.asarray([[2.0, 1.0, 0.5], [3.0, 0.0, 1.0], [1.0, 1.0, 1.0]]), "kpc")
    ts = u.Q(jnp.asarray([0.0, 100.0, 200.0]), "Myr")

    got = getattr(pot, method)(xyz, ts)
    want = jnp.stack(
        [u.ustrip(got.unit, getattr(pot, method)(xyz[i], ts[i])) for i in range(3)]
    )
    assert jnp.array_equal(u.ustrip(got.unit, got), want)


def test_a_constant_multipole_keeps_every_other_potential_s_shapes() -> None:
    """Shapes must match the rest of the library, composites included.

    REGRESSION: `vectorize_method` was added to `_potential` and `_density`
    to make a *grid* build accept a batched ``t``. It did, but the decorator
    broadcasts the *loop* dims of ``xyz`` and ``t``, which no other galax
    potential's `_potential` does -- so a **constant**-parameter multipole,
    a path this feature does not touch at all, started answering ``(7,)``
    where `HernquistPotential` answers ``()``.

    `AbstractCompositePotential._potential` stacks its components rather
    than broadcasting them, so that divergence was not a cosmetic shape
    difference: any composite holding a multipole raised ``TypeError:
    Cannot concatenate arrays with different numbers of dimensions``. The
    whole 2913-test potential suite passed with it broken, because nothing
    asserted a shape here.

    Pinned against `HernquistPotential` rather than against literals, so it
    tracks the library's convention instead of a snapshot of it.
    """
    src = HernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(5.0, "kpc"), units="galactic"
    )
    mp = MultipoleProfilePotential.from_potential(
        src,
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e2, "kpc"),
        n_r=64,
        l_max=2,
        symmetry="plane_reflection",
    )
    other = HernquistPotential(
        m_tot=u.Q(1e11, "Msun"), r_s=u.Q(3.0, "kpc"), units="galactic"
    )

    xyz1 = u.Q(jnp.asarray([1.0, 2.0, 3.0]), "kpc")
    xyz7 = u.Q(jnp.ones((7, 3)), "kpc")
    t7 = u.Q(jnp.linspace(0.0, 100.0, 7), "Myr")
    t0 = u.Q(0.0, "Myr")

    for xyz, t in ((xyz1, t7), (xyz1, t0), (xyz7, t0), (xyz7, t7)):
        assert jnp.shape(mp.potential(xyz, t).value) == jnp.shape(
            other.potential(xyz, t).value
        ), (jnp.shape(xyz.value), jnp.shape(t.value))

    # The composite stacks its components, so a shape divergence is a crash.
    comp = CompositePotential(a=mp, b=other)
    assert jnp.shape(comp.potential(xyz1, t7).value) == ()
    assert jnp.shape(comp.potential(xyz7, t0).value) == (7,)

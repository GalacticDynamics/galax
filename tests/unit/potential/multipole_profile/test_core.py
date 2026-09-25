"""Tests for the `MultipoleProfilePotential` class."""

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.params as gpp
from galax.potential import Symmetry
from galax.potential._src.base import default_constants
from galax.potential._src.builtin.multipole_profile.build import build_expansion
from galax.potential._src.builtin.multipole_profile.core import (
    AbstractMultipoleProfilePotential,
    MultipoleProfilePotential,
)
from galax.potential._src.harmonic import (
    default_angular_resolution,
    lm_keys,
)

G_GALACTIC = 4.498502265137816e-12
"""G in kpc^3 / (Msun Myr^2)."""

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
    """A built expansion cannot track a time-varying source.

    Better to refuse than to hand back a potential silently inconsistent with
    its own density. Lifting this needs
    https://github.com/GalacticDynamics/galax/issues/849
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
    with pytest.raises(ValueError, match="time-dependent"):
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

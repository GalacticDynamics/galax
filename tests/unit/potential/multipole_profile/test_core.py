"""Tests for the `MultipoleProfilePotential` class."""

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.params as gpp
from galax.potential._src.builtin.multipole_profile.core import (
    AbstractMultipoleProfilePotential,
    MultipoleProfilePotential,
)
from galax.potential._src.builtin.multipole_profile.funcs import (
    build_expansion,
    radial_grid,
)
from galax.potential._src.builtin.multipole_profile.project import (
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
    r = radial_grid(256, jnp.asarray(1e-2), jnp.asarray(1e4))
    coeffs = build_expansion(
        rho, r, 0, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(G_GALACTIC)
    )
    return MultipoleProfilePotential(
        r_knots=u.Q(r, "kpc"),
        phi_lm=u.Q(coeffs["phi_lm"], "kpc2 / Myr2"),
        dphi_lm=u.Q(coeffs["dphi_lm"], "kpc2 / Myr2"),
        rho_residual_lm=u.Q(coeffs["rho_residual_lm"], "Msun / kpc3"),
        drho_residual_lm=u.Q(coeffs["drho_residual_lm"], "Msun / kpc3"),
        rho_amplitude=u.Q(coeffs["rho_amplitude"], "Msun / kpc3"),
        rho_alpha=u.Q(coeffs["rho_alpha"], ""),
        l_max=0,
        lm_keys=keys,
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
    assert pot.symmetry == "spherical"


def test_params_carries_every_coefficient() -> None:
    """`_params` is the contract between the class and the functions."""
    p = _hernquist_potential()._params(u.Q(0.0, "Gyr"))
    assert set(p) == {
        "r_knots",
        "phi_lm",
        "dphi_lm",
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


def test_from_density_defaults_angular_resolution_from_l_max() -> None:
    def rho(xyz, t):
        return jnp.exp(-jnp.linalg.norm(xyz, axis=-1))

    pot = MultipoleProfilePotential.from_density(
        rho,
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e2, "kpc"),
        n_r=32,
        l_max=4,
        symmetry="triaxial",
        units="galactic",
    )
    assert pot.l_max == 4
    assert pot.symmetry == "triaxial"
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

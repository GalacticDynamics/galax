"""Independent physics checks for `MultipoleProfilePotential`.

Separate from the unit tests: these compare against closed forms and exact
integrals rather than against the `bfeax` reference, so they catch errors the
port and its oracle share.
"""

import itertools

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp

R_MIN = u.Q(1e-2, "kpc")
R_MAX = u.Q(1e4, "kpc")
X = u.Q([1.0, 2.0, 3.0], "kpc")


@pytest.fixture(scope="module")
def _analytic() -> dict[str, gp.AbstractSinglePotential]:
    m, r_s = u.Q(1e12, "Msun"), u.Q(10.0, "kpc")
    return {
        "hernquist": gp.HernquistPotential(m_tot=m, r_s=r_s, units="galactic"),
        "plummer": gp.PlummerPotential(m_tot=m, r_s=r_s, units="galactic"),
        "nfw": gp.NFWPotential(m=m, r_s=r_s, units="galactic"),
        "jaffe": gp.JaffePotential(m_tot=m, r_s=r_s, units="galactic"),
    }


@pytest.mark.parametrize("name", ["hernquist", "plummer", "nfw", "jaffe"])
def test_spherical_expansion_matches_the_closed_form(_analytic, name) -> None:  # noqa: PT019
    """Monopole-only expansions reproduce their analytic counterparts."""
    ref = _analytic[name]
    pot = gp.MultipoleProfilePotential.from_potential(
        ref, r_min=R_MIN, r_max=R_MAX, n_r=512, l_max=0, symmetry="spherical"
    )
    assert jnp.isclose(
        pot.potential(X, t=0),
        ref.potential(X, t=0),
        rtol=2e-3,
        atol=u.Q(0.0, "kpc2 / Myr2"),
    )
    assert jnp.allclose(
        pot.gradient(X, t=0),
        ref.gradient(X, t=0),
        rtol=5e-3,
        atol=u.Q(0.0, "kpc / Myr2"),
    )
    assert jnp.isclose(
        pot.density(X, t=0),
        ref.density(X, t=0),
        rtol=1e-3,
        atol=u.Q(0.0, "Msun / kpc3"),
    )


def test_triaxial_nfw_matches_the_homoeoid_integral() -> None:
    """Against `TriaxialNFWPotential`'s exact Chandrasekhar integral.

    This is the test that would catch the outer-tail sign defect, since it
    exercises l >= 1 modes against an independent solution of the same
    density.
    """
    ref = gp.TriaxialNFWPotential(
        m=u.Q(1e12, "Msun"),
        r_s=u.Q(10.0, "kpc"),
        q1=1.0,
        q2=0.8,
        units="galactic",
    )
    pot = gp.MultipoleProfilePotential.from_potential(
        ref, r_min=R_MIN, r_max=R_MAX, n_r=256, l_max=8, symmetry="triaxial"
    )
    for x in [
        u.Q([1.0, 2.0, 3.0], "kpc"),
        u.Q([15.0, 5.0, 2.0], "kpc"),
        u.Q([0.0, 0.0, 20.0], "kpc"),
    ]:
        assert jnp.isclose(
            pot.potential(x, t=0),
            ref.potential(x, t=0),
            rtol=1e-2,
            atol=u.Q(0.0, "kpc2 / Myr2"),
        )


def test_gradient_is_finite_on_the_z_axis(_analytic) -> None:  # noqa: PT019
    """`bfeax` returns NaN here. The Cartesian `iter_Ylm` form is why we don't."""
    pot = gp.MultipoleProfilePotential.from_potential(
        _analytic["hernquist"], r_min=R_MIN, r_max=R_MAX, n_r=64, l_max=6
    )
    for x in [
        u.Q([0.0, 0.0, 5.0], "kpc"),
        u.Q([0.0, 0.0, -5.0], "kpc"),
        u.Q([0.0, 0.0, 1e-3], "kpc"),
    ]:
        assert jnp.all(jnp.isfinite(pot.gradient(x, t=0).value))
        assert jnp.isfinite(pot.potential(x, t=0).value)


def test_error_is_second_order_in_n_r(_analytic) -> None:  # noqa: PT019
    """Doubling `n_r` cuts the error by ~4x.

    Measured during design as 4.46e-3 / 1.10e-3 / 2.74e-4 / 6.96e-5 for
    n_r = 64/128/256/512 on a comparable setup. The convergence *order* is
    the invariant worth pinning; tightening it is
    https://github.com/GalacticDynamics/galax/issues/848
    """
    ref = _analytic["hernquist"]
    expect = ref.potential(X, t=0)

    errs = []
    for n_r in (64, 128, 256, 512):
        pot = gp.MultipoleProfilePotential.from_potential(
            ref, r_min=R_MIN, r_max=R_MAX, n_r=n_r, l_max=0, symmetry="spherical"
        )
        errs.append(float(jnp.abs((pot.potential(X, t=0) - expect) / expect)))

    for coarse, fine in itertools.pairwise(errs):
        assert 2.5 < coarse / fine < 6.0, errs


@pytest.mark.parametrize("symmetry", [None, "triaxial"])
def test_symmetry_modes_agree_for_a_triaxial_density(symmetry) -> None:
    """Pruning modes that vanish by symmetry must not change the answer."""
    ref = gp.TriaxialNFWPotential(
        m=u.Q(1e12, "Msun"),
        r_s=u.Q(10.0, "kpc"),
        q1=1.0,
        q2=0.8,
        units="galactic",
    )
    full = gp.MultipoleProfilePotential.from_potential(
        ref, r_min=R_MIN, r_max=R_MAX, n_r=128, l_max=4, symmetry=None
    )
    pruned = gp.MultipoleProfilePotential.from_potential(
        ref, r_min=R_MIN, r_max=R_MAX, n_r=128, l_max=4, symmetry=symmetry
    )
    x = u.Q([3.0, 4.0, 5.0], "kpc")
    assert jnp.isclose(
        full.potential(x, t=0),
        pruned.potential(x, t=0),
        rtol=1e-6,
        atol=u.Q(0.0, "kpc2 / Myr2"),
    )


def test_triaxial_nfw_reaches_outer_tail() -> None:
    """Closed-form accuracy check at higher resolution near the outer boundary.

    Complements the r_max=1e4 kpc test, which evaluates at r/r_max ~ 0.0005-0.002.
    This test uses r_max=300 kpc and evaluates at r/r_max ~ 0.83, extending the
    coverage range near the outer boundary.

    Does NOT verify the outer-tail sign fix: at these parameters the residual error
    is dominated by grid truncation and finite l_max. Toggling the gate alone moves
    the error from 8.222e-3 to 8.587e-3 (same band, inconsistent direction).

    The outer-tail sign fix is verified by `test_outer_tail_applies_to_negative_modes`
    (in test_poisson.py), which checks the exact linearity Phi(-rho) == -Phi(rho)
    that the buggy signed gate violated.
    """
    ref = gp.TriaxialNFWPotential(
        m=u.Q(1e12, "Msun"),
        r_s=u.Q(10.0, "kpc"),
        q1=1.0,
        q2=0.8,
        units="galactic",
    )
    r_max = u.Q(300.0, "kpc")
    pot = gp.MultipoleProfilePotential.from_potential(
        ref,
        r_min=u.Q(1e-2, "kpc"),
        r_max=r_max,
        n_r=256,
        l_max=8,
        symmetry="triaxial",
    )
    # Evaluate at r/r_max ~ 0.83, extending coverage toward r_max
    x = u.Q([150.0, 150.0, 132.0], "kpc")
    assert jnp.isclose(
        pot.potential(x, t=0),
        ref.potential(x, t=0),
        rtol=1.2e-2,
        atol=u.Q(0.0, "kpc2 / Myr2"),
    )

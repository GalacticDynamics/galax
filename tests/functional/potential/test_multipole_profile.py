"""Independent physics checks for `MultipoleProfilePotential`.

Separate from the unit tests: these compare against closed forms and exact
integrals, so they catch errors a comparison against another implementation
of the same algorithm would share.
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

    This is a closed-form accuracy check. Does NOT verify the outer-tail
    sign fix; at these parameters the residual error is dominated by grid
    truncation and finite l_max, so toggling the gate changes the error
    inconsistently and imperceptibly.

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
    pot = gp.MultipoleProfilePotential.from_potential(
        ref, r_min=R_MIN, r_max=R_MAX, n_r=256, l_max=8, symmetry="plane_reflection"
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
    """A theta/phi harmonic gives NaN here; the Cartesian `iter_Ylm` does not.

    On the z-axis ``atan2``'s gradient is 0/0 and ``acos``'s is infinite, so
    autodiff through an angular form returns NaN for every term. `iter_Ylm`
    evaluates from the Cartesian direction, which is polynomial in x and y
    and so smooth there.
    """
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


def test_symmetry_modes_agree_for_a_triaxial_density() -> None:
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
        ref, r_min=R_MIN, r_max=R_MAX, n_r=128, l_max=4, symmetry="plane_reflection"
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
        symmetry="plane_reflection",
    )
    # Evaluate at r/r_max ~ 0.83, extending coverage toward r_max
    x = u.Q([150.0, 150.0, 132.0], "kpc")
    assert jnp.isclose(
        pot.potential(x, t=0),
        ref.potential(x, t=0),
        rtol=1.2e-2,
        atol=u.Q(0.0, "kpc2 / Myr2"),
    )


def test_outer_tail_stays_bounded_and_signed() -> None:
    r"""Outside the grid the potential continues as a fitted power law.

    This replaces a regression test that pinned the *unguarded* edge-cubic
    extrapolation, which grew without bound and flipped sign. Same potential
    and same radii, so the two are directly comparable:

    ======= ========== ========== =========
    r/r_max  edge cubic      fitted     truth
    ======= ========== ========== =========
       1.17     0.0247     0.0020    (rel. error)
       1.67     0.7312     0.0098
       2.33     4.1020 +   0.0152
       3.33    15.8019 +   0.0194
      10.00        --       0.0259
    ======= ========== ========== =========

    ``+`` marks where the edge cubic came back *positive* for a bound
    (negative) potential. The tail never does: it is :math:`P_1 x^v + \ldots`
    with :math:`P_1` the boundary value, so it cannot cross zero going
    outward for a monopole.

    The bound here is 0.05, loose enough not to trip on quadrature noise and
    tight enough that any regression to the edge cubic (0.73 by
    :math:`1.7 r_\max`) fails it. Accuracy is capped by the spline's natural
    end condition, tracked at
    https://github.com/GalacticDynamics/galax/issues/858
    """
    ref = gp.HernquistPotential(
        m_tot=u.Q(1e12, "Msun"),
        r_s=u.Q(10.0, "kpc"),
        units="galactic",
    )
    r_max = 300.0
    pot = gp.MultipoleProfilePotential.from_potential(
        ref,
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(r_max, "kpc"),
        n_r=128,
        l_max=0,
        symmetry="spherical",
    )

    for ratio in (1.17, 1.67, 2.33, 3.33, 10.0):
        x = u.Q([r_max * ratio, 0.0, 0.0], "kpc")
        phi = pot.potential(x, t=0)
        truth = ref.potential(x, t=0)

        assert phi < u.Q(0.0, "kpc2 / Myr2"), (
            f"the potential must stay bound outside the grid, "
            f"got {phi} at r/r_max = {ratio}"
        )
        err = jnp.abs((phi - truth) / truth)
        assert err < 0.05, f"rel. error {err} at r/r_max = {ratio}"


def test_inner_tail_stays_finite() -> None:
    """The same continuation runs inward, where the old cubic also blew up."""
    ref = gp.HernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(10.0, "kpc"), units="galactic"
    )
    pot = gp.MultipoleProfilePotential.from_potential(
        ref,
        r_min=u.Q(1.0, "kpc"),
        r_max=u.Q(300.0, "kpc"),
        n_r=128,
        l_max=0,
        symmetry="spherical",
    )
    # Two decades below r_min, where a cubic in log r has plenty of room.
    x = u.Q([1e-2, 0.0, 0.0], "kpc")
    phi = pot.potential(x, t=0)
    assert jnp.isfinite(phi.value)
    assert phi < u.Q(0.0, "kpc2 / Myr2")


def test_potential_and_gradient_are_sane_at_the_exact_origin(_analytic) -> None:  # noqa: PT019
    r"""The inner power-law tail also fixed the origin.

    This replaces a regression test that pinned :math:`\nabla\Phi(0) =`
    **NaN** -- a single NaN poisons an entire vmapped batch of orbits, so it
    was the worse of the two extrapolation defects. That test asked whoever
    fixed it to state the new behaviour deliberately; this is that statement.

    At exactly :math:`\vec{x} = 0` the direction is still undefined, but
    :math:`\log r` does not underflow: `safe_vector_norm` floors :math:`r` at
    :math:`\sqrt{\mathrm{tiny}}`, so :math:`\log r` is a large finite
    negative (-354 in float64) rather than :math:`-\infty`.
    `eval_log_spline_asympt` then evaluates the inner tail on a clamped
    :math:`L = \min(\log r - \log r_\min, 0)`, so the branch is finite for
    every query radius either way. The monopole tail tends to a finite central value:
    :math:`\Phi(0) = -0.4530` against a true :math:`-0.4499` (0.69%), where
    the bare edge cubic gave :math:`\approx -2 \times 10^4`. The gradient
    comes back exactly zero, which is also the analytic answer.

    The *density* is not continued and is still meaningless here (~1e163);
    that limitation is recorded in the class docstring.
    """
    ref = _analytic["hernquist"]
    pot = gp.MultipoleProfilePotential.from_potential(
        ref, r_min=R_MIN, r_max=R_MAX, n_r=64, l_max=0, symmetry="spherical"
    )
    origin = u.Q([0.0, 0.0, 0.0], "kpc")

    phi, truth = pot.potential(origin, t=0), ref.potential(origin, t=0)
    assert jnp.isfinite(phi.value)
    assert jnp.abs((phi - truth) / truth) < 0.02

    grad = pot.gradient(origin, t=0).value
    assert jnp.all(jnp.isfinite(grad)), f"the origin must not produce NaN, got {grad}"
    assert jnp.allclose(
        grad, 0.0
    ), f"expected a zero gradient at the centre, got {grad}"
    assert jnp.all(jnp.isfinite(ref.gradient(origin, t=0).value))


def test_default_angular_resolution_controls_aliasing() -> None:
    """The oversampled default must actually buy the accuracy it advertises.

    A flattened halo is not band-limited, so under the minimal rule
    (``l_max + 2``, ``2 l_max + 1``) the power above ``l_max`` aliases into
    the retained modes. Measured here against a converged (60, 61) rule at
    the same ``l_max`` -- aliasing alone, truncation held fixed -- the worst
    relative error is 6.0e-4 with the current default and 1.7e-2 under the
    minimal rule, so the 3e-3 bound below separates the two.

    Without this, reverting `default_angular_resolution` to the minimal rule
    fails only the test that restates its formula.
    """
    ref = gp.TriaxialNFWPotential(
        m=u.Q(1e12, "Msun"),
        r_s=u.Q(10.0, "kpc"),
        q1=1.0,
        q2=0.4,
        units="galactic",
    )
    kw = {"r_min": R_MIN, "r_max": R_MAX, "n_r": 256, "l_max": 4}
    default = gp.MultipoleProfilePotential.from_potential(ref, **kw)
    converged = gp.MultipoleProfilePotential.from_potential(
        ref, n_theta=60, n_phi=61, **kw
    )
    for x in [
        u.Q([1.0, 2.0, 3.0], "kpc"),
        u.Q([0.0, 0.0, 20.0], "kpc"),
        u.Q([15.0, 5.0, 2.0], "kpc"),
    ]:
        assert jnp.isclose(
            default.potential(x, t=0),
            converged.potential(x, t=0),
            rtol=3e-3,
            atol=u.Q(0.0, "kpc2 / Myr2"),
        )


def test_symmetry_modes_agree_for_an_axisymmetric_density() -> None:
    """Pruning to ``m = 0`` must not change an axisymmetric answer.

    The counterpart of `test_symmetry_modes_agree_for_a_triaxial_density`
    for the case a "z-rotation plus z-reflection" reading would get wrong, by
    also dropping odd ``l``. `MiyamotoNagaiPotential` is axisymmetric but not
    spherical, so
    the pruned and unpruned builds differ in every mode but the answer.
    """
    ref = gp.MiyamotoNagaiPotential(
        m_tot=u.Q(1e11, "Msun"),
        a=u.Q(3.0, "kpc"),
        b=u.Q(0.3, "kpc"),
        units="galactic",
    )
    kw = {"r_min": R_MIN, "r_max": u.Q(1e3, "kpc"), "n_r": 128, "l_max": 4}
    full = gp.MultipoleProfilePotential.from_potential(ref, symmetry=None, **kw)
    pruned = gp.MultipoleProfilePotential.from_potential(
        ref, symmetry="zrotation", **kw
    )
    assert len(pruned.lm_keys) == 5  # (0,0) .. (4,0), odd l included
    for x in [u.Q([3.0, 4.0, 5.0], "kpc"), u.Q([8.0, 0.0, 1.0], "kpc")]:
        assert jnp.isclose(
            full.potential(x, t=0),
            pruned.potential(x, t=0),
            rtol=1e-6,
            atol=u.Q(0.0, "kpc2 / Myr2"),
        )

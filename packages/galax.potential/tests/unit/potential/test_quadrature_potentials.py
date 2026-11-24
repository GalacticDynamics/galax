r"""Tests for the three quadrature potentials.

The potential of `AxisymmetricGaussianPotential`, `TriaxialGaussianPotential`
and `TriaxialNFWPotential` is Chandrasekhar's integral, which the kernels
evaluate by Gauss-Legendre quadrature. The references here are `mpmath.quad`
of the same integral in its original form, the one over :math:`\tau \in
[0, \infty)`, at 30 digits.
"""

import mpmath as mp
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp

G = "galactic"
M = u.Q(1e12, "Msun")
RS = u.Q(5, "kpc")
# q1 != q2, so that a swap of the two shows.
CASES = {
    "axisymmetricgaussian": gp.AxisymmetricGaussianPotential(
        m_tot=M, r_s=RS, q2=0.7, units=G
    ),
    "triaxialgaussian": gp.TriaxialGaussianPotential(
        m_tot=M, r_s=RS, q1=0.9, q2=0.6, units=G
    ),
    "triaxialnfw": gp.TriaxialNFWPotential(m=M, r_s=RS, q1=0.9, q2=0.6, units=G),
}
# A direction with all three components different, as a fraction of the radius.
DIRECTION = (2.0 / 7, 3.0 / 7, 6.0 / 7)
# Inside, near and far: 0.1, 1 and 30 r_s.
RADII_IN_RS = (0.1, 1.0, 30.0)


def mp_potential(name, p, xyz, dps=30):
    """Chandrasekhar's integral over tau in [0, inf) at ``dps`` digits."""
    with mp.workdps(dps):
        G_, rs = mp.mpf(float(p["G"])), mp.mpf(float(p["r_s"]))
        q1 = mp.mpf(float(p["q1"])) if "q1" in p else mp.mpf(1)
        q2 = mp.mpf(float(p["q2"]))
        x, y, z = (mp.mpf(float(c)) for c in xyz)
        if name == "triaxialnfw":
            rho0 = mp.mpf(float(p["m"])) / (4 * mp.pi * rs**3)

            def dpsi(xi2):
                return 2 / (1 + mp.sqrt(xi2))

        else:
            m = mp.mpf(float(p["m_tot"]))
            rho0 = m / (q1 * q2 * (2 * mp.pi) ** mp.mpf("1.5") * rs**3)

            def dpsi(xi2):
                return 2 * mp.exp(-xi2 / 2)

        def f(tau):
            xi2 = x**2 / (1 + tau) + y**2 / (q1**2 + tau) + z**2 / (q2**2 + tau)
            xi2 /= rs**2
            return dpsi(xi2) / mp.sqrt((1 + tau) * (q1**2 + tau) * (q2**2 + tau))

        r = mp.sqrt(x**2 + y**2 + z**2) / rs
        edges = [0, *(r**2 * mp.mpf(10) ** k for k in range(-6, 7, 2)), mp.inf]
        edges = sorted(set(edges))
        integral = mp.quad(f, edges)
        return -mp.pi * G_ * q1 * q2 * rho0 * rs**2 * integral


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("r_in_rs", RADII_IN_RS)
def test_potential_matches_mpmath(name, r_in_rs):
    """The order-50 kernel equals `mp.quad` of the integral, inside, near and far."""
    pot = CASES[name]
    module = getattr(gp, name)
    p = pot.params(0.0)
    xyz = jnp.asarray(DIRECTION) * r_in_rs * 5.0  # kpc
    got = float(module.potential(p, xyz))
    want = float(mp_potential(name, p, xyz))
    # Measured: <= 3e-14 at these radii, see the class docstrings.
    np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize("order", [10, 20])
@pytest.mark.parametrize("name", CASES)
def test_integration_order_is_used_and_static(name, order):
    """``integration_order`` reaches the quadrature, in the class and the module."""
    pot = CASES[name]
    module = getattr(gp, name)
    p = pot.params(0.0)
    xyz = jnp.asarray(DIRECTION) * 20.0 * 5.0  # 20 r_s

    low = module.potential(p, xyz, integration_order=order)
    high = module.potential(p, xyz)
    assert module.potential(p, xyz, integration_order=50) == high
    assert low != high

    pot_low = type(pot)(
        **{
            f: getattr(pot, f)
            for f in ("m_tot", "m", "r_s", "q1", "q2")
            if hasattr(pot, f)
        },
        units=G,
        integration_order=order,
    )
    np.testing.assert_allclose(pot_low.potential(u.Q(xyz, "kpc"), 0.0).value, low)
    # The class gradient and the module gradient agree at the same order.
    np.testing.assert_allclose(
        pot_low.gradient(xyz, 0.0),
        module.gradient(p, xyz, integration_order=order),
        rtol=1e-10,
    )
    # `hessian` and `gradient` take the keyword, and the order changes them.
    for fn in (module.gradient, module.hessian):
        assert not np.array_equal(fn(p, xyz, integration_order=order), fn(p, xyz))


def test_spherical_limits():
    """For ``q1 = q2 = 1`` the three are the spherical Gaussian and NFW."""
    xyz = jnp.array([1.0, 2.0, 3.0])
    for name, sph in (
        ("axisymmetricgaussian", gp.GaussianPotential(m_tot=M, r_s=RS, units=G)),
        ("triaxialgaussian", gp.GaussianPotential(m_tot=M, r_s=RS, units=G)),
        ("triaxialnfw", gp.NFWPotential(m=M, r_s=RS, units=G)),
    ):
        pot = type(CASES[name])(
            **{f: getattr(sph, f) for f in ("m_tot", "m", "r_s") if hasattr(sph, f)},
            units=G,
        )
        np.testing.assert_allclose(
            pot.potential(xyz, 0.0), sph.potential(xyz, 0.0), rtol=1e-12
        )
        np.testing.assert_allclose(
            pot.density(xyz, 0.0), sph.density(xyz, 0.0), rtol=1e-12
        )


@pytest.mark.parametrize("q2", [0.7, 1.5])
def test_axisymmetric_central_potential(q2):
    """``Phi(0) = -sqrt(2/pi) G m / r_s * f(q2)``: arcsin for q2 < 1, arcsinh above."""
    pot = gp.AxisymmetricGaussianPotential(m_tot=M, r_s=RS, q2=q2, units=G)
    p = pot.params(0.0)
    if q2 < 1:
        e = np.sqrt(1 - q2**2)
        f = np.arcsin(e) / e
    else:
        e = np.sqrt(q2**2 - 1)
        f = np.arcsinh(e) / e
    want = -np.sqrt(2 / np.pi) * p["G"] * p["m_tot"] / p["r_s"] * f
    got = gp.axisymmetricgaussian.potential(p, jnp.zeros(3))
    np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize("name", ["axisymmetricgaussian", "triaxialgaussian"])
def test_gaussian_density_integrates_to_the_mass(name):
    """The sum of the density over a fine box equals ``m_tot``, for any axis ratio.

    The trapezoid rule is spectrally accurate for a Gaussian that is negligible at
    the edge of the box (9 widths along the widest axis, 4 nodes per width).
    """
    p = CASES[name].params(0.0)
    h = float(p["r_s"]) / 4
    n = 9 * 4 * 2 + 1
    x = (np.arange(n) - n // 2) * h
    xyz = jnp.asarray(np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1))
    mass = float(jnp.sum(getattr(gp, name).density(p, xyz))) * h**3
    np.testing.assert_allclose(mass, float(p["m_tot"]), rtol=1e-8)

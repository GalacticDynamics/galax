r"""Tests of the module functions of the three quadrature potentials.

`axisymmetricgaussian`, `triaxialgaussian` and `triaxialnfw` have a potential
that is a one-dimensional integral. The reference here is that integral in
Chandrasekhar's form over :math:`\tau \in [0, \infty)`, evaluated by mpmath,
which is independent of the substitution :math:`\tau = 1 / s^2 - 1` and of the
Gauss-Legendre nodes that the kernels use.
"""

import jax
import mpmath as mp
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp

MP_DPS = 30  # read by the autouse fixture in conftest.py

M, RS = u.Q(1e12, "Msun"), u.Q(8.0, "kpc")
Q1, Q2 = 0.9, 0.6

# name -> (potential, module, Delta psi / (rho0 r_s^2) as a function of xi)
CASES = {
    "axisymmetricgaussian": (
        gp.AxisymmetricGaussianPotential(m_tot=M, r_s=RS, q2=Q2, units="galactic"),
        gp.axisymmetricgaussian,
        lambda xi: 2 * mp.exp(-(xi**2) / 2),
    ),
    "triaxialgaussian": (
        gp.TriaxialGaussianPotential(m_tot=M, r_s=RS, q1=Q1, q2=Q2, units="galactic"),
        gp.triaxialgaussian,
        lambda xi: 2 * mp.exp(-(xi**2) / 2),
    ),
    "triaxialnfw": (
        gp.TriaxialNFWPotential(m=M, r_s=RS, q1=Q1, q2=Q2, units="galactic"),
        gp.triaxialnfw,
        lambda xi: 2 / (1 + xi),
    ),
}

# 0.1, 1 and 30 scale radii, in directions off the axes (kpc).
POINTS = [(0.8, -0.3, 0.5), (4.0, 5.0, -3.0), (160.0, -120.0, 90.0)]


def chandrasekhar(name, xyz, p):
    """``Phi`` from the integral over ``tau``, in mpmath."""
    x, y, z = (mp.mpf(float(v)) for v in xyz)
    rs, G = mp.mpf(float(p["r_s"])), mp.mpf(float(p["G"]))
    q1 = mp.mpf(float(p["q1"])) if "q1" in p else mp.mpf(1)
    q2 = mp.mpf(float(p["q2"]))
    rho0 = mp.mpf(float(module_rho0(name, p)))
    dpsi = CASES[name][2]

    def f(tau):
        xi = (
            mp.sqrt(x**2 / (1 + tau) + y**2 / (q1**2 + tau) + z**2 / (q2**2 + tau)) / rs
        )
        return dpsi(xi) / mp.sqrt((1 + tau) * (q1**2 + tau) * (q2**2 + tau))

    return -mp.pi * G * q1 * q2 * rho0 * rs**2 * mp.quad(f, [0, 1, 10, 100, mp.inf])


def module_rho0(name, p):
    return CASES[name][1].rho0_of_m(p)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("xyz", POINTS, ids=lambda xyz: f"r{np.linalg.norm(xyz):.0f}")
def test_potential_matches_mpmath(name, xyz):
    pot, module, _ = CASES[name]
    p = pot.params(0.0)
    got = module.potential(p, jnp.array(xyz))
    np.testing.assert_allclose(got, float(chandrasekhar(name, xyz, p)), rtol=1e-12)


def test_central_density_from_mass():
    """``rho0_of_m`` gives the central density of each profile."""
    for name, (pot, module, _) in CASES.items():
        p = pot.params(0.0)
        q1 = float(p["q1"]) if "q1" in p else 1.0
        rs3 = float(p["r_s"]) ** 3
        if name == "triaxialnfw":
            want = float(p["m"]) / (4 * np.pi * rs3)
        else:
            want = float(p["m_tot"]) / (q1 * float(p["q2"]) * (2 * np.pi) ** 1.5 * rs3)
        np.testing.assert_allclose(module.rho0_of_m(p), want, rtol=1e-14)


@pytest.mark.parametrize("name", CASES)
def test_integration_order_reaches_the_kernel(name):
    """The keyword changes the quadrature, and the class passes its own value."""
    pot, module, _ = CASES[name]
    cls, p, xyz = type(pot), pot.params(0.0), jnp.array(POINTS[-1])
    fields = {
        f.name: getattr(pot, f.name)
        for f in __import__("dataclasses").fields(pot)
        if f.name not in ("units", "constants", "integration_order")
    }
    coarse = cls(**fields, units="galactic", integration_order=4)

    low = module.potential(p, xyz, integration_order=4)
    default = module.potential(p, xyz)
    assert default == module.potential(p, xyz, integration_order=50)
    assert abs(low - default) > 1e-8 * abs(default)
    np.testing.assert_allclose(coarse.potential(xyz, 0.0), low, rtol=1e-14)

    # The derived functions take the keyword too.
    for fn in (module.gradient, module.hessian):
        a, b = fn(p, xyz, integration_order=4), fn(p, xyz)
        assert not np.allclose(a, b, rtol=1e-8, atol=0)
    np.testing.assert_allclose(
        coarse.gradient(xyz, 0.0), module.gradient(p, xyz, integration_order=4)
    )


@pytest.mark.parametrize("name", CASES)
def test_default_order_error_is_small(name):
    """Order 50 is accurate to 1e-12 at 0.1, 1 and 30 scale radii (see above)."""
    pot, module, _ = CASES[name]
    p = pot.params(0.0)
    for xyz in POINTS:
        ref = float(chandrasekhar(name, xyz, p))
        got = float(module.potential(p, jnp.array(xyz)))
        assert abs(got - ref) <= 1e-12 * abs(ref)


@pytest.mark.parametrize("name", CASES)
def test_batch_and_empty_batch(name):
    pot, module, _ = CASES[name]
    xyz = jnp.array(POINTS)
    batched = pot.potential(xyz, 0.0)
    per_point = jax.vmap(lambda x: module.potential(pot.params(0.0), x))(xyz)
    np.testing.assert_allclose(batched, per_point, rtol=1e-14)
    assert pot.potential(jnp.zeros((0, 3)), 0.0).shape == (0,)

"""Tests of the module functions of the three multipole potentials."""

import jax
import mpmath as mp
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp

MP_DPS = 30  # read by the autouse fixture in conftest.py

M, RS = u.Q(1e12, "Msun"), u.Q(8.0, "kpc")
XYZ = (3.0, -4.0, 5.0)  # kpc, off every axis and plane

LM = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 2), (3, 3)]


def unit_coeffs(l_max, l, m):
    """Coefficients that are zero except for one entry, which is 1."""
    return jnp.zeros((l_max + 1, l_max + 1)).at[l, m].set(1.0)


def harmonic(l, m, xyz):
    """``Y_lm`` at ``xyz`` from mpmath, as a complex number."""
    x, y, z = (mp.mpf(v) for v in xyz)
    r = mp.sqrt(x**2 + y**2 + z**2)
    return mp.spherharm(l, m, mp.acos(z / r), mp.atan2(y, x))


@pytest.mark.parametrize(("l", "m"), LM)
@pytest.mark.parametrize("which", ["S", "T"])
@pytest.mark.parametrize("kind", ["inner", "outer"])
def test_single_term_matches_mpmath(kind, which, l, m):
    """One coefficient gives ``s**l`` (or ``s**-(l+1)``) times ``Re`` or ``Im Y_lm``."""
    l_max = 3
    ones, zeros = unit_coeffs(l_max, l, m), jnp.zeros((l_max + 1, l_max + 1))
    cls = gp.MultipoleInnerPotential if kind == "inner" else gp.MultipoleOuterPotential
    module = gp.multipoleinner if kind == "inner" else gp.multipoleouter
    pot = cls(
        m_tot=M,
        r_s=RS,
        l_max=l_max,
        Slm=ones if which == "S" else zeros,
        Tlm=ones if which == "T" else zeros,
        units="galactic",
    )
    p = pot.params(0.0)
    r = float(np.linalg.norm(XYZ))
    s = r / float(p["r_s"])
    power = s**l if kind == "inner" else s ** -(l + 1)
    y = harmonic(l, m, XYZ)
    angular = float(y.real if which == "S" else y.imag)
    want = float(p["G"] * p["m_tot"] / p["r_s"]) * power * angular

    np.testing.assert_allclose(module.potential(p, jnp.array(XYZ)), want, rtol=1e-12)
    np.testing.assert_allclose(pot.potential(jnp.array(XYZ), 0.0), want, rtol=1e-12)


def test_combined_is_inner_plus_outer():
    """``MultipolePotential`` is the sum of an inner and an outer expansion."""
    key = jax.random.key(0)
    s_in, t_in, s_out, t_out = (
        jnp.tril(jax.random.normal(k, (4, 4))) for k in jax.random.split(key, 4)
    )
    kw = {"m_tot": M, "r_s": RS, "l_max": 3, "units": "galactic"}
    both = gp.MultipolePotential(ISlm=s_in, ITlm=t_in, OSlm=s_out, OTlm=t_out, **kw)
    inner = gp.MultipoleInnerPotential(Slm=s_in, Tlm=t_in, **kw)
    outer = gp.MultipoleOuterPotential(Slm=s_out, Tlm=t_out, **kw)
    xyz = jnp.array(XYZ)

    np.testing.assert_allclose(
        gp.multipole.potential(both.params(0.0), xyz),
        gp.multipoleinner.potential(inner.params(0.0), xyz)
        + gp.multipoleouter.potential(outer.params(0.0), xyz),
        rtol=1e-12,
    )


@pytest.mark.parametrize("l_max", [0, 1, 4])
def test_l_max_comes_from_the_coefficient_shape(l_max):
    """A kernel given ``(l_max + 1, l_max + 1)`` coefficients equals the class."""
    s = jnp.tril(jnp.arange(1.0, (l_max + 1) ** 2 + 1).reshape(l_max + 1, -1)) / 10
    pot = gp.MultipoleInnerPotential(
        m_tot=M, r_s=RS, l_max=l_max, Slm=s, Tlm=0.5 * s, units="galactic"
    )
    xyz = jnp.array(XYZ)
    np.testing.assert_allclose(
        gp.multipoleinner.potential(pot.params(0.0), xyz),
        pot.potential(xyz, 0.0),
        rtol=1e-14,
    )
    # Dropping the highest degree from the coefficients changes the value.
    if l_max > 0:
        p = pot.params(0.0)
        short = {**p, "Slm": p["Slm"][:l_max, :l_max], "Tlm": p["Tlm"][:l_max, :l_max]}
        assert gp.multipoleinner.potential(short, xyz) != gp.multipoleinner.potential(
            p, xyz
        )


def test_density_and_laplacian_are_exactly_zero():
    pot = gp.MultipoleOuterPotential(
        m_tot=M,
        r_s=RS,
        l_max=1,
        Slm=unit_coeffs(1, 1, 1),
        Tlm=unit_coeffs(1, 0, 0),
        units="galactic",
    )
    p, xyz = pot.params(0.0), jnp.array(XYZ)
    for module in (gp.multipole, gp.multipoleinner, gp.multipoleouter):
        assert module.density(p, xyz) == 0.0
        assert module.laplacian(p, xyz) == 0.0
        assert module.density(p, xyz).shape == ()
    # One shared function, so the three modules agree on the density.
    assert gp.multipoleinner.density is gp.multipole.density
    assert gp.multipoleouter.density is gp.multipole.density


def test_gradient_on_the_z_axis_is_finite():
    """The Cartesian direction keeps the gradient and Hessian finite on the pole."""
    pot = gp.MultipoleInnerPotential(
        m_tot=M,
        r_s=RS,
        l_max=3,
        Slm=jnp.tril(jnp.ones((4, 4))),
        Tlm=jnp.tril(jnp.ones((4, 4))),
        units="galactic",
    )
    p, xyz = pot.params(0.0), jnp.array([0.0, 0.0, 6.0])
    grad = gp.multipoleinner.gradient(p, xyz)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.all(jnp.isfinite(gp.multipoleinner.hessian(p, xyz)))
    np.testing.assert_allclose(
        grad, jax.grad(gp.multipoleinner.potential, 1)(p, xyz), rtol=1e-12
    )

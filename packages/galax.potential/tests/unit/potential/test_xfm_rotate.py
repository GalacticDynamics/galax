"""Tests for `galax.potential.RotatingPotential`."""

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp

OMEGA = u.Q(52.2, "km/(s kpc)")
PHI_B = u.Q(25, "deg")
MONARI = {"R0": u.Q(8.0, "kpc"), "v0": u.Q(220.0, "km/s"), "Rb": u.Q(3.5, "kpc")}

# Points inside and outside the bar radius (3.5 kpc), at several times.
_XYZ = u.Q(np.random.default_rng(0).uniform(-8, 8, size=(32, 3)) * [1, 1, 0.3], "kpc")
_T = u.Q(np.linspace(-1_000, 1_000, 32), "Myr")


def _close(a: u.Quantity, b: u.Quantity, rtol: float = 1e-12) -> None:
    a, b = np.asarray(a.ustrip(b.unit)), np.asarray(b.ustrip(b.unit))
    np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * np.abs(b).max())


@pytest.fixture
def rotating() -> gp.RotatingPotential:
    """Return a static Monari bar, rotated by the transform."""
    static = gp.MonariEtAl2016BarPotential(
        **MONARI, Omega=u.Q(0.0, "km/(s kpc)"), phi_b=u.Q(0, "deg"), units="galactic"
    )
    return gp.RotatingPotential(static, Omega=OMEGA, phi_0=PHI_B)


@pytest.mark.parametrize("method", ["potential", "gradient", "density", "hessian"])
def test_reproduces_monari_pattern_speed(
    rotating: gp.RotatingPotential, method: str
) -> None:
    """Rotating a static bar equals the bar's own ``Omega`` and ``phi_b``."""
    monari = gp.MonariEtAl2016BarPotential(
        **MONARI, Omega=OMEGA, phi_b=PHI_B, units="galactic"
    )
    got = getattr(rotating, method)(_XYZ, _T)
    want = getattr(monari, method)(_XYZ, _T)
    assert got.shape == want.shape
    _close(got, want)


@pytest.mark.parametrize("method", ["potential", "gradient", "density", "hessian"])
def test_z_axisymmetric_base_is_unchanged(method: str) -> None:
    base = gp.MiyamotoNagaiPotential(
        m_tot=u.Q(1e11, "Msun"), a=u.Q(3, "kpc"), b=u.Q(0.3, "kpc"), units="galactic"
    )
    pot = gp.RotatingPotential(base, Omega=OMEGA, phi_0=PHI_B)
    assert pot.symmetry is base.symmetry
    _close(getattr(pot, method)(_XYZ, _T), getattr(base, method)(_XYZ, _T))


def test_symmetry_of_a_bar_is_none(rotating: gp.RotatingPotential) -> None:
    triaxial = gp.TriaxialHernquistPotential(
        m_tot=u.Q(1e10, "Msun"), r_s=u.Q(1, "kpc"), q1=0.8, q2=0.6, units="galactic"
    )
    assert triaxial.symmetry is gp.Symmetry.PLANE_REFLECTION
    pot = gp.RotatingPotential(triaxial, Omega=OMEGA)
    assert pot.symmetry is gp.Symmetry.NONE


def test_vmap_matches_batch(rotating: gp.RotatingPotential) -> None:
    """Mapping over points and times equals the batched call."""
    ul, ut = rotating.units["length"], rotating.units["time"]
    xyz, t = _XYZ.ustrip(ul), _T.ustrip(ut)

    for method in ("_potential", "_gradient", "_density", "_hessian"):
        mapped = jax.jit(jax.vmap(getattr(rotating, method)))(xyz, t)
        batched = getattr(rotating, method)(xyz, t)
        _close(u.Q(mapped, ""), u.Q(batched, ""), rtol=1e-14)


def test_grad_wrt_omega_matches_finite_difference() -> None:
    """``jit(grad)`` with respect to the pattern speed, vs central differences."""
    bar = gp.LongMuraliBarPotential(
        m_tot=u.Q(1e10, "Msun"),
        a=u.Q(4, "kpc"),
        b=u.Q(1, "kpc"),
        c=u.Q(1, "kpc"),
        alpha=u.Q(0, "rad"),
        units="galactic",
    )
    xyz = u.Q([[3.0, 1.0, 0.2], [-1.0, 4.0, -0.5]], "kpc")
    t = u.Q([150.0, -400.0], "Myr")

    def phi(omega: jax.Array) -> jax.Array:
        pot = gp.RotatingPotential(bar, Omega=u.Q(omega, "1/Myr"))
        return pot.potential(xyz, t).value.sum()

    def acc(omega: jax.Array) -> jax.Array:
        pot = gp.RotatingPotential(bar, Omega=u.Q(omega, "1/Myr"))
        return pot.acceleration(xyz, t).value

    omega0, h = 0.04, 1e-6
    for f, grad in ((phi, jax.grad), (acc, jax.jacfwd)):
        got = jax.jit(grad(f))(omega0)
        want = (f(omega0 + h) - f(omega0 - h)) / (2 * h)
        assert jnp.all(got != 0)
        np.testing.assert_allclose(got, want, rtol=1e-6)

"""Tests of `TabulatedRadialBasisPotential`.

The oracle is `SCFPotential`: a table of its own Hernquist radial functions must
reproduce it, so the tabulated class is checked against an independent analytic
implementation for every (n, l, m) term, including m > 0 and the sin terms.
"""

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential.params import CustomParameter
from galax.potential.scf import phi_nl

G = 4.498502151469554e-12  # kpc^3 / (Msun Myr^2)
M_TOT, R_S = 1e12, 10.0
NMAX, LMAX = 2, 2
R_KNOTS = jnp.logspace(-2.5, 3.5, 600)  # kpc


def _coeffs() -> tuple[jax.Array, jax.Array]:
    key1, key2 = jax.random.split(jax.random.key(0))
    s = 0.1 * jax.random.normal(key1, (NMAX + 1, LMAX + 1, LMAX + 1))
    t = 0.1 * jax.random.normal(key2, (NMAX + 1, LMAX + 1, LMAX + 1))
    tri = jnp.tril(jnp.ones((LMAX + 1, LMAX + 1)))  # only m <= l is meaningful
    return (s * tri).at[0, 0, 0].set(1.0), t * tri.at[:, 0].set(
        0.0
    )  # no sin term at m = 0


def _table() -> u.Quantity:
    radial = (
        G * M_TOT / R_S * phi_nl(NMAX, LMAX, R_KNOTS / R_S)
    )  # (nmax+1, lmax+1, n_r)
    return u.Q(jnp.moveaxis(radial, -1, 0), "kpc2 / Myr2")


@pytest.fixture(scope="module")
def pair() -> tuple[gp.TabulatedRadialBasisPotential, gp.SCFPotential]:
    S, T = _coeffs()
    tab = gp.TabulatedRadialBasisPotential.from_knots(
        u.Q(R_KNOTS, "kpc"), _table(), S, T, units="galactic"
    )
    scf = gp.SCFPotential(
        m_tot=u.Q(M_TOT, "Msun"), r_s=u.Q(R_S, "kpc"), Snlm=S, Tnlm=T, units="galactic"
    )
    return tab, scf


POINTS = u.Q(
    jnp.array(
        [
            [8.0, 1.0, 2.0],
            [-3.0, 4.0, 5.0],
            [0.3, -0.2, 0.1],
            [40.0, -60.0, 25.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, -50.0],
        ]
    ),  # the last two on the z-axis
    "kpc",
)


def test_matches_scf_potential(pair) -> None:
    tab, scf = pair
    got, expect = tab.potential(POINTS, 0.0), scf.potential(POINTS, 0.0)
    assert jnp.allclose(got, expect, rtol=1e-7, atol=u.Q(0.0, expect.unit))


def _derivs(pot, f, x):
    return jnp.stack([f(lambda q: pot._potential(q, 0.0))(xi) for xi in x])


def test_matches_scf_gradient(pair) -> None:
    """Autodiff gradient; the spline's derivative is good to O(h^3) in h."""
    tab, scf = pair
    x = jnp.asarray(POINTS.value)
    got, expect = _derivs(tab, jax.grad, x), _derivs(scf, jax.grad, x)
    assert jnp.all(jnp.isfinite(got))  # including on the z-axis
    err = jnp.linalg.norm(got - expect, axis=1) / jnp.linalg.norm(expect, axis=1)
    assert jnp.all(err < 1e-4), err


def test_matches_scf_hessian(pair) -> None:
    """Autodiff hessian; the spline's second derivative is good to O(h^2)."""
    tab, scf = pair
    x = jnp.asarray(POINTS.value[:4])
    got, expect = _derivs(tab, jax.hessian, x), _derivs(scf, jax.hessian, x)
    err = jnp.linalg.norm((got - expect).reshape(len(x), -1), axis=1) / jnp.linalg.norm(
        expect.reshape(len(x), -1), axis=1
    )
    assert jnp.all(err < 1e-2), err


def test_time_dependent_coefficients() -> None:
    """Coefficients from a `CustomParameter` give the potential of those at t."""
    S, T = _coeffs()

    def ramp(t, s0):  # linear in time: s0 at t = 0, 2 s0 at t = 1 Gyr
        return s0 * (1.0 + u.ustrip("Gyr", t))

    tdep = gp.TabulatedRadialBasisPotential.from_knots(
        u.Q(R_KNOTS, "kpc"),
        _table(),
        CustomParameter(func=ramp, args=(S,)),
        T,
        units="galactic",
    )
    snap = gp.TabulatedRadialBasisPotential.from_knots(
        u.Q(R_KNOTS, "kpc"), _table(), 1.5 * S, T, units="galactic"
    )
    t = u.Q(0.5, "Gyr")
    got, expect = tdep.potential(POINTS, t), snap.potential(POINTS, t)
    assert jnp.allclose(got, expect, rtol=1e-12, atol=u.Q(0.0, expect.unit))


def test_outside_the_knots_is_held(pair) -> None:
    """Beyond the outermost knot the radial functions stay at their edge values."""
    tab, _ = pair
    far = u.Q(jnp.array([[1e5, 0.0, 0.0], [2e5, 0.0, 0.0]]), "kpc")
    p = tab.potential(far, 0.0)
    assert jnp.all(jnp.isfinite(p.value))
    assert jnp.isclose(p[0], p[1], rtol=1e-12, atol=u.Q(0.0, p.unit))
    assert jnp.all(jnp.isfinite(tab.potential(u.Q([0.0, 0.0, 0.0], "kpc"), 0.0).value))


def test_shape_checks() -> None:
    S, T = _coeffs()
    with pytest.raises(ValueError, match="Snlm and Tnlm"):
        gp.TabulatedRadialBasisPotential.from_knots(
            u.Q(R_KNOTS, "kpc"), _table(), S, T[:, :2, :2], units="galactic"
        )
    with pytest.raises(ValueError, match="u_nl must have shape"):
        gp.TabulatedRadialBasisPotential.from_knots(
            u.Q(R_KNOTS, "kpc"), _table()[:, :2], S, T, units="galactic"
        )

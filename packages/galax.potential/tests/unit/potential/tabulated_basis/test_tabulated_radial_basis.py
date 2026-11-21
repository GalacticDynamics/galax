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
    )  # (n_max+1, l_max+1, n_r)
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


def test_every_term_matches_scf() -> None:
    """Each (n, l, m) term alone, cos and sin, against `SCFPotential`."""
    x = POINTS[:4]
    zero = jnp.zeros((NMAX + 1, LMAX + 1, LMAX + 1))
    for n in range(NMAX + 1):
        for l in range(LMAX + 1):
            for m in range(l + 1):
                for which in ("S", "T") if m > 0 else ("S",):
                    c = zero.at[n, l, m].set(1.0)
                    S, T = (c, zero) if which == "S" else (zero, c)
                    tab = gp.TabulatedRadialBasisPotential.from_knots(
                        u.Q(R_KNOTS, "kpc"), _table(), S, T, units="galactic"
                    )
                    scf = gp.SCFPotential(
                        m_tot=u.Q(M_TOT, "Msun"),
                        r_s=u.Q(R_S, "kpc"),
                        Snlm=S,
                        Tnlm=T,
                        units="galactic",
                    )
                    got, expect = tab.potential(x, 0.0), scf.potential(x, 0.0)
                    scale = jnp.max(jnp.abs(expect.value))
                    err = jnp.max(jnp.abs((got - expect).value)) / scale
                    assert err < 1e-6, (n, l, m, which, err)


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
    x = jnp.asarray(POINTS.value)  # including the z-axis
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


def test_outside_the_knots(pair) -> None:
    """Beyond the last knot the terms fall as r^-(l+1); inside the first, as r^l."""
    tab, scf = pair
    r_max = float(R_KNOTS[-1])
    grad = lambda pot, x: jax.grad(lambda q: pot._potential(q, 0.0))(x)
    xhat = jnp.array([0.6, 0.0, 0.8])
    # continuous across the last knot (to the Hernquist vs Kepler tail, ~2 r_s / r)
    inside, outside = (
        grad(tab, (1 - 1e-6) * r_max * xhat),
        grad(tab, (1 + 1e-6) * r_max * xhat),
    )
    assert jnp.allclose(inside, outside, rtol=1e-2)
    # and close to the true (SCF) force well beyond it
    far = 3 * r_max * xhat
    assert jnp.allclose(grad(tab, far), grad(scf, far), rtol=1e-2)
    # finite potential and force at the origin
    origin = jnp.zeros(3)
    assert jnp.isfinite(tab._potential(origin, 0.0))
    assert jnp.all(jnp.isfinite(grad(tab, origin)))


def test_a_time_per_position() -> None:
    """With time-dependent coefficients, one time per position."""
    S, T = _coeffs()
    tdep = gp.TabulatedRadialBasisPotential.from_knots(
        u.Q(R_KNOTS, "kpc"),
        _table(),
        CustomParameter(func=lambda t, s0: s0 * (1.0 + u.ustrip("Gyr", t)), args=(S,)),
        T,
        units="galactic",
    )
    ts = u.Q(jnp.linspace(0.0, 1.0, POINTS.shape[0]), "Gyr")
    batched = tdep.potential(POINTS, ts)
    one_by_one = jnp.stack(
        [tdep.potential(x, ti).value for x, ti in zip(POINTS, ts, strict=True)]
    )
    assert jnp.allclose(batched.value, one_by_one, rtol=1e-12)


def test_knot_checks() -> None:
    S, T = _coeffs()
    bad = R_KNOTS.at[3].set(R_KNOTS[2])  # not strictly increasing
    with pytest.raises(ValueError, match="strictly increasing"):
        gp.TabulatedRadialBasisPotential.from_knots(
            u.Q(bad, "kpc"), _table(), S, T, units="galactic"
        )


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

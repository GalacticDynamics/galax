"""The dense-interpolation wrapper around an orbit."""

import equinox as eqx
import pytest

import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp


@pytest.fixture
def interpolant() -> gd.orbit.PhaseSpaceInterpolation:
    """Return a dense Kepler orbit's interpolant."""
    pot = gp.KeplerPotential(m_tot=u.Q(1e12, "Msun"), units="galactic")
    w0 = gc.PhaseSpacePosition(q=u.Q([8.0, 0, 0], "kpc"), p=u.Q([0.0, 220, 0], "km/s"))
    orbit = gd.evaluate_orbit(pot, w0, u.Q([0.0, 100.0], "Myr"), dense=True)
    return orbit.interpolant


def test_exposes_the_underlying_interpolation(interpolant) -> None:
    """The three pass-through properties onto the vectorized interpolation."""
    assert interpolant.batch_shape == ()
    assert interpolant.batch_ndim == 0
    assert interpolant.scalar_interpolation is not None


def test_evaluates_inside_the_bounds(interpolant) -> None:
    """A time within the integration window interpolates."""
    w = interpolant.evaluate(u.Q(50.0, "Myr"))
    assert w.q.shape == ()


def test_rejects_a_time_outside_the_bounds(interpolant) -> None:
    """The guard is a quantity comparison, which must be stripped for jax."""
    with pytest.raises(eqx.EquinoxRuntimeError, match="Time out of bounds"):
        interpolant.evaluate(u.Q(1000.0, "Myr"))

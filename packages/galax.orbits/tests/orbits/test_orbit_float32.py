"""An orbit from float32 initial conditions is float32, also with x64 on."""

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.orbits as go
import galax.potential as gp


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_orbit_dtype_follows_input(dtype: object) -> None:
    pot = gp.HernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(5, "kpc"), units="galactic"
    )
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q(jnp.asarray([8.0, 0, 1], dtype=dtype), "kpc"),
        p=u.Q(jnp.asarray([0.0, 0.2, 0], dtype=dtype), "kpc/Myr"),
        t=u.Q(jnp.asarray(0.0, dtype=dtype), "Myr"),
    )
    ts = u.Q(jnp.linspace(0, 100, 5, dtype=dtype), "Myr")
    orbit = go.compute_orbit(pot, w0, ts)
    assert orbit.q.x.dtype == dtype
    assert orbit.p.x.dtype == dtype
    assert orbit.t.dtype == dtype


def test_mixed_dtypes_promote_together() -> None:
    """float32 positions with float64 velocities integrate in float64."""
    pot = gp.HernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(5, "kpc"), units="galactic"
    )
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q(jnp.asarray([8.0, 0, 1], dtype=jnp.float32), "kpc"),
        p=u.Q(jnp.asarray([0.0, 0.2, 0], dtype=jnp.float64), "kpc/Myr"),
        t=u.Q(jnp.asarray(0.0, dtype=jnp.float32), "Myr"),
    )
    ts = u.Q(jnp.linspace(0, 100, 5, dtype=jnp.float32), "Myr")
    assert go.compute_orbit(pot, w0, ts).q.x.dtype == jnp.float64

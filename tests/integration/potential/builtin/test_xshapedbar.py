"""Orbits in `galax.potential.XShapedBarPotential`."""

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.orbits as go
import galax.potential as gp


def test_orbit_conserves_energy() -> None:
    """A bulge orbit in the static bar plus a halo conserves its energy."""
    bar = gp.XShapedBarPotential.build(l_max=8)
    pot = gp.CompositePotential(
        bar=bar,
        halo=gp.NFWPotential(m=u.Q(5e11, "Msun"), r_s=u.Q(15, "kpc"), units="galactic"),
    )
    w0 = gc.PhaseSpacePosition(
        q=u.Q([2.0, 0.5, 0.3], "kpc"), p=u.Q([10.0, 120.0, 20.0], "km/s")
    )
    t = u.Q(jnp.linspace(0, 1, 101), "Gyr")
    orbit = go.compute_orbit(pot, w0, t)

    energy = gp.total_energy(pot, orbit).ustrip("kpc2/Myr2")
    assert jnp.all(jnp.isfinite(orbit.q.x.value))
    assert jnp.max(jnp.abs(energy / energy[0] - 1)) < 1e-6

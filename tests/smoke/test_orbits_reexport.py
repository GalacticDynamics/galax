"""galax.dynamics re-exports the objects that live in galax.orbits."""

import galax.dynamics as gd
import galax.dynamics._src.utils as gdu
import galax.orbits as go
import galax.orbits.utils as gou


def test_fields_identity():
    assert gd.AbstractOrbitField is go.AbstractOrbitField
    assert gd.HamiltonianField is go.HamiltonianField
    assert gd.fields.HamiltonianField is go.HamiltonianField
    assert gdu.parse_to_t_y is gou.parse_to_t_y


def test_orbit_identity():
    for name in ("Orbit", "OrbitSolver", "compute_orbit"):
        assert getattr(gd, name) is getattr(go, name)
    for name in ("Orbit", "OrbitSolver", "PhaseSpaceInterpolation", "compute_orbit"):
        assert getattr(gd.orbit, name) is getattr(go, name)
    assert gd.plot.plot_components is go.plot.plot_components


def test_plot_dispatch_after_move():
    import matplotlib as mpl

    mpl.use("Agg")

    import jax.numpy as jnp

    import unxt as u

    import galax.coordinates as gc
    import galax.potential as gp

    pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q([8.0, 0, 0], "kpc"), p=u.Q([0, 220, 0], "km/s"), t=u.Q(0, "Myr")
    )
    orbit = go.compute_orbit(pot, w0, u.Q(jnp.linspace(0, 100, 5), "Myr"))
    assert isinstance(orbit, gd.Orbit)
    assert orbit.plot() is not None


def test_experimental_identity():
    import galax.dynamics._src.experimental as gdx
    import galax.orbits._src.experimental as gox

    assert gdx.Leapfrog is gox.Leapfrog
    assert gdx.integrate_orbit is gox.integrate_orbit
    assert gdx.SymplecticSolverT is gox.SymplecticSolverT

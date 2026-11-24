"""galax.dynamics re-exports the objects that live in galax.orbits."""

import galax.dynamics as gd
import galax.orbits as go


def test_fields_identity():
    assert gd.AbstractOrbitField is go.AbstractOrbitField
    assert gd.HamiltonianField is go.HamiltonianField
    assert gd.fields.HamiltonianField is go.HamiltonianField


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

    assert not hasattr(gdx, "Leapfrog")  # use galax.fields.Leapfrog
    assert gdx.integrate_orbit is gox.integrate_orbit
    assert gdx.SymplecticSolverT is gox.SymplecticSolverT


def test_experimental_submodules():
    import galax.dynamics as gd
    import galax.orbits._src.experimental.integrate as goi

    assert gd.experimental.integrate is goi
    assert not hasattr(gd.experimental, "leapfrog")


def test_kinematics_identity():
    import galax.dynamics as gd
    import galax.dynamics._src.api as gda
    import galax.orbits as go

    assert gd.omega is go.omega is gda.omega
    assert gd.specific_angular_momentum is go.specific_angular_momentum
    assert gda.specific_angular_momentum is go.specific_angular_momentum


def test_integrate_identity():
    import galax.dynamics as gd
    import galax.dynamics._src.legacy as gdl
    import galax.orbits.integrate as goi

    assert gd.evaluate_orbit is goi.evaluate_orbit is gdl.evaluate_orbit
    assert gd.integrate.Integrator is goi.Integrator is gdl.Integrator
    assert (
        gd.integrate.InterpolatedPhaseSpaceCoordinate
        is goi.InterpolatedPhaseSpaceCoordinate
    )
    assert gdl.default_integrator is goi.default_integrator
    assert gd.integrate.PhaseSpaceInterpolation is goi.PhaseSpaceInterpolation
    assert gd.integrate.parse_time_specification is goi.parse_time_specification
    assert gdl.InterpolatedPhaseSpaceCoordinate is goi.InterpolatedPhaseSpaceCoordinate


def test_custom_types_alias_identity():
    import galax.dynamics.custom_types as gdt
    import galax.orbits.custom_types as got

    for name in ("BtQuSz3", "SzTime", "OptArgs"):
        assert getattr(gdt, name) is getattr(got, name)


def test_experimental_public_identity():
    import galax.dynamics._src.experimental as gdx
    import galax.orbits._src.experimental as gox_private
    import galax.orbits.experimental as gox

    assert gox.integrate_orbit is gox_private.integrate_orbit is gdx.integrate_orbit
    assert gox.SymplecticSolverT is gdx.SymplecticSolverT
    assert gdx.integrate is gox_private.integrate

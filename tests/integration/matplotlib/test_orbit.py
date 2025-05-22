"""Test the `galax.dynamics.orbit` package contents."""

import pytest
from matplotlib.figure import Figure

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp


@pytest.fixture
def potential() -> gp.KeplerPotential:
    """Kepler potential fixture."""
    return gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")


@pytest.fixture
def w0() -> gc.PhaseSpacePosition:
    """Phase space position fixture."""
    return gc.PhaseSpacePosition(
        q=u.Quantity([8.0, 0.0, 0.5], "kpc"),
        p=u.Quantity([0.0, 220.0, 0.0], "km/s"),
    )


@pytest.fixture
def orbit(potential: gp.AbstractPotential, w0: gc.PhaseSpacePosition) -> gd.Orbit:
    """Orbit fixture."""
    ts = u.Quantity(jnp.linspace(0.0, 70, 1000), "Myr")
    orb: gd.Orbit = gd.evaluate_orbit(potential, w0, ts)
    return orb


# =============================================================================


@pytest.mark.mpl_image_compare(deterministic=True)
def test_orbit_plot_all_components(orbit: gd.Orbit) -> Figure:
    """Test plotting all components of an orbit in a Kepler potential."""
    axes = orbit.plot()
    return axes[0].figure


@pytest.mark.mpl_image_compare(deterministic=True)
def test_orbit_plot(orbit: gd.Orbit) -> Figure:
    """Test plotting an orbit in a Kepler potential."""
    ax = orbit.plot(x="x", y="y")

    return ax.figure


@pytest.mark.mpl_image_compare(deterministic=True)
def test_orbit_plot_represent_as(orbit: gd.Orbit) -> Figure:
    """Test plotting an orbit in a Kepler potential."""
    ax = orbit.plot(x="rho", y="d_z", vector_representation=cx.vecs.CylindricalPos)

    return ax.figure


@pytest.mark.mpl_image_compare(deterministic=True)
def test_orbit_plot_scatter(orbit: gd.Orbit) -> Figure:
    """Test plotting an orbit in a Kepler potential."""
    ax = orbit.plot(x="x", y="y", plot_function="scatter")

    return ax.figure


@pytest.mark.mpl_image_compare(deterministic=True)
def test_orbit_plot_time_color(orbit: gd.Orbit) -> Figure:
    """Test plotting an orbit in a Kepler potential."""
    ax = orbit.plot(x="x", y="y", plot_function="scatter", c="orbit.t")

    return ax.figure


def test_orbit_no_attribute(orbit: gd.Orbit) -> None:
    """Test failed plot."""
    with pytest.raises(AttributeError):
        orbit.plot(x="z", y="not_an_attribute")

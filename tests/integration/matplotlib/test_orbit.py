"""Test the `galax.dynamics.orbit` package contents."""

import pytest
from matplotlib.figure import Figure

import quaxed.numpy as jnp
from unxt import Quantity

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp


@pytest.mark.mpl_image_compare(deterministic=True)
def test_orbit_plot() -> Figure:
    """Test plotting an orbit in a Kepler potential."""
    pot = gp.KeplerPotential(
        m_tot=Quantity(1e12, "Msun"),
        units="galactic",
    )
    w0 = gc.PhaseSpacePosition(
        q=Quantity([8.0, 0.0, 0.0], "kpc"),
        p=Quantity([0.0, 220.0, 0.0], "km/s"),
    )
    ts = Quantity(jnp.linspace(0.0, 100.0, 1000), "Myr")
    orbit = gd.evaluate_orbit(pot, w0, ts)

    ax = orbit.plot(x="x", y="y")

    return ax.figure

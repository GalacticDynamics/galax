"""Test the `galax.dynamics.orbit` package contents."""

from itertools import combinations
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import optype.numpy as onp
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
def field(potential: gp.KeplerPotential) -> gd.fields.HamiltonianField:
    """Hamiltonian field fixture."""
    return gd.fields.HamiltonianField(potential)


@pytest.fixture
def solver() -> gd.DynamicsSolver:
    """Dynamics solver fixture."""
    return gd.DynamicsSolver()


FigAx23: TypeAlias = tuple[Figure, onp.Array[tuple[Literal[2], Literal[3]], Axes]]


@pytest.fixture
def sixaxfig() -> FigAx23:
    """Six axis figure fixture."""
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    return fig, axs


# =============================================================================


@pytest.mark.mpl_image_compare
def test_solution_plot(
    solver: gd.DynamicsSolver, field: gd.fields.HamiltonianField, sixaxfig: FigAx23
) -> Figure:
    """Test plotting an orbit in a Kepler potential."""
    # Solve the dynamical system
    w0 = gc.PhaseSpacePosition(
        q=u.Quantity([8.0, 0.0, 0.5], "kpc"),
        p=u.Quantity([0.0, 220.0, 0.0], "km/s"),
        t=u.Quantity(0.0, "Gyr"),
    )
    tf = u.Quantity(200, "Myr")
    saveat = jnp.linspace(w0.t, tf, 1000)
    soln = solver.solve(field, w0, tf, saveat=saveat)
    q, p = soln.ys

    # Plot the solution
    fig, axs = sixaxfig
    fig.suptitle("Phase space solution")

    usys = field.units
    labels = np.empty((2, 3), dtype="<U17")
    labels[0, :] = [f"{x} [{usys['length']}]" for x in ["x", "y", "z"]]
    labels[1, :] = [f"{v} [{usys['speed']}]" for v in [r"$v_x$", r"$v_y$", r"$v_z$"]]

    for ax, (i, j) in zip(axs[0, :], combinations(range(3), 2), strict=True):
        ax.plot(q[..., i], q[..., j])
        ax.set(xlabel=labels[0, i], ylabel=labels[0, j])
    for ax, (i, j) in zip(axs[1, :], combinations(range(3), 2), strict=True):
        ax.plot(p[..., i], p[..., j])
        ax.set(xlabel=labels[1, i], ylabel=labels[1, j])

    fig.tight_layout()

    return fig

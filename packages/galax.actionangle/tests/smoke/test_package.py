"""Testing :mod:`galax.actionangle` module."""

import subprocess
import sys

from galax import actionangle


def test_all() -> None:
    """Test the `galax.actionangle` API."""
    assert set(actionangle.__all__) == {
        "ActionsAnglesFrequencies",
        "isochrone_actions_angles_freqs",
        "staeckel_fudge_delta",
    }
    assert all(hasattr(actionangle, name) for name in actionangle.__all__)


_FRESH = """
import jax.numpy as jnp
import unxt as u
import galax.coordinates as gc
import galax.potential as gp
from galax.actionangle import isochrone_actions_angles_freqs as f
from galax.actionangle import staeckel_fudge_delta as fd

pot = gp.IsochronePotential(m_tot=u.Q(1e11, "Msun"), r_s=u.Q(2.0, "kpc"),
                            units="galactic")
q, p = jnp.array([[8.0, 0, 1], [5, 1, 0]]), jnp.array([[0, 0.15, 0.01]] * 2)
res = f(pot, q, p, jnp.asarray(0.0))
assert res.actions.shape == res.angles.shape == res.freqs.shape == (2, 3)
res = f(pot, u.Q(q, "kpc"), u.Q(p, "kpc/Myr"), u.Q(0.0, "Myr"))
assert res.actions.shape == (2, 3) and str(res.angles.unit) == "rad"
assert fd(pot, q, jnp.asarray(0.0)).shape == (2,)
w = gc.PhaseSpaceCoordinate(q=u.Q(q, "kpc"), p=u.Q(p, "kpc/Myr"), t=u.Q(0.0, "Myr"))
assert fd(pot, w).shape == (2,)
"""


def test_overloads_registered_on_import() -> None:
    """The overloads register on a plain import (fresh interpreter)."""
    subprocess.run([sys.executable, "-c", _FRESH], check=True)  # noqa: S603

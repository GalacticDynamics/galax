"""Isochrone actions, angles and frequencies against agama's.

For an ``Isochrone`` potential agama's action finder is the analytic one
(``actions::evalIsochrone``), so both sides are closed forms. They use the same
angle origins and conventions: no offset is needed, and the angles agree to
round-off. The one exception is ``theta_z`` of a planar orbit, which is
undefined: agama sets it to 0 and galax follows gala (``psi = pi/2``), so it
is not compared.

agama orders every triplet ``(r, z, phi)``; galax orders them ``(r, phi, z)``.
"""

from types import ModuleType

import numpy as np
import pytest

import unxt as u

import galax.potential as gp
from ...actionangle.optional_deps import OptDeps
from galax.actionangle import isochrone_actions_angles_freqs

pytestmark = [
    pytest.mark.agama,
    pytest.mark.skipif(not OptDeps.AGAMA.installed, reason="requires agama"),
]

M, B = 1e11, 2.0  # Msun, kpc

# (kpc, km/s). v_c(8 kpc) = 178.3 km/s.
POINTS = {
    "near-circular": ([8.0, 0.0, 0.0], [1.8, 178.1, 3.6]),
    "eccentric": ([5.0, 3.0, 2.0], [80.0, 60.0, 40.0]),
    "near-radial": ([3.0, 1.0, -0.5], [150.0, -20.0, 10.0]),
    "polar": ([6.0, 0.0, 1.0], [30.0, 0.0, 150.0]),  # L_z = 0 exactly
    "near-polar": ([6.0, 0.0, 1.0], [30.0, 1e-3, 150.0]),
    "retrograde": ([7.0, -2.0, 1.5], [20.0, -150.0, 30.0]),
    "planar": ([4.0, 2.0, 0.0], [-50.0, 120.0, 0.0]),
}

# Measured worst cases (agama 1.0), each tolerance >= 4x above them:
# - actions: relative 2.9e-15, except where J is a small difference of large
#   terms. J_r goes through E, so its absolute precision is ~ eps L (see
#   `isochrone_actions_angles_freqs`): the near-circular J_r is off by 6.7e-13
#   kpc km/s = 2.1 eps L (1.1e-11 relative), its J_z by 1.6e-13.
# - frequencies: relative 1.1e-15.
# - angles: 9.3e-14 rad for the near-circular theta_r (ill-defined as e -> 0),
#   1.3e-15 rad for the other orbits.
ACTIONS_RTOL, ACTIONS_ATOL_EPS_L = 1e-14, 10  # atol in units of eps L
FREQS_RTOL = 5e-15
ANGLES_ATOL = 4e-13

_GALAX_ORDER = [0, 2, 1]  # agama's (r, z, phi) -> galax's (r, phi, z)


def _agama_aaf(agama: ModuleType, finder: str, xv: np.ndarray) -> tuple:
    pot = agama.Potential(type="Isochrone", mass=M, scaleRadius=B)
    out = (
        agama.ActionFinder(pot)(xv, angles=True)
        if finder == "ActionFinder"
        else agama.actions(pot, xv, angles=True)
    )
    return tuple(a[:, _GALAX_ORDER] for a in out)


@pytest.mark.parametrize("finder", ["ActionFinder", "actions"])
def test_isochrone_matches_agama(agama: ModuleType, finder: str) -> None:
    xv = np.array([np.r_[x, v] for x, v in POINTS.values()])
    aJ, aT, aO = _agama_aaf(agama, finder, xv)

    # agama's G exceeds galax's (astropy's) by 1.49e-10 relative. Scale the
    # mass so GM is the same on both sides and the comparison sees round-off.
    g = gp.IsochronePotential(m_tot=1, r_s=1, units="galactic").constants["G"]
    m = M * agama.G / u.ustrip("kpc km2 / (s2 Msun)", g)
    pot = gp.IsochronePotential(
        m_tot=u.Q(m, "Msun"), r_s=u.Q(B, "kpc"), units="galactic"
    )
    res = isochrone_actions_angles_freqs(
        pot, u.Q(xv[:, :3], "kpc"), u.Q(xv[:, 3:], "km/s"), u.Q(0, "Myr")
    )
    J = np.asarray(u.ustrip("kpc km/s", res.actions))
    T = np.asarray(u.ustrip("rad", res.angles))
    O = np.asarray(u.ustrip("rad km / (s kpc)", res.freqs))

    dT = np.angle(np.exp(1j * (T - aT)))  # wrapped to (-pi, pi]
    dT[list(POINTS).index("planar"), 2] = 0  # theta_z undefined: see module doc
    eps_L = np.finfo(float).eps * np.linalg.norm(np.cross(xv[:, :3], xv[:, 3:]), axis=1)
    for i, name in enumerate(POINTS):
        np.testing.assert_allclose(
            J[i],
            aJ[i],
            rtol=ACTIONS_RTOL,
            atol=ACTIONS_ATOL_EPS_L * eps_L[i],
            err_msg=name,
        )
        np.testing.assert_allclose(O[i], aO[i], rtol=FREQS_RTOL, err_msg=name)
        np.testing.assert_allclose(dT[i], 0, atol=ANGLES_ATOL, err_msg=name)

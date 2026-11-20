"""O2GF actions against galpy's ``actionAngleIsochroneApprox``.

galpy's method (Bovy 2014) is independent of O2GF, so the actions agree only to
within the errors of the two methods, and the tolerance comes from measuring
them. The orbits are those of the agama torus round trip in
``tests/integration/agama/test_o2gf_vs_agama.py``, in MilkyWayPotential2022.
galpy builds the same potential from its own Miyamoto-Nagai, NFW and Hernquist
classes, in natural units with galax's G, and integrates the orbit itself.

A second test compares O2GF on an isochrone orbit, with a fitted toy, against
galpy's exact ``actionAngleIsochrone``.

galpy calls pass ``quantity=False``, since a local galpyrc can turn on astropy
output. These tests run in CI's "Check Interoperability" job, which installs
galpy through the ``all`` extra (``uv sync --extra all --group test-all``).
"""

import numpy as np
import pytest

import unxt as u

import galax.potential as gp
from ...actionangle._helpers import mw2022_parts, o2gf_from_point
from ...actionangle.optional_deps import OptDeps

pytestmark = [
    pytest.mark.skipif(not OptDeps.GALPY.installed, reason="requires galpy"),
    # Informational galpy warnings. Its PyPI wheel ships without the torus C
    # extension, which this test does not use, and importing galpy.actionAngle
    # says so. It also says when it integrates in C. Other galpy warnings, such
    # as an orbit too short for its fit, still fail the test.
    pytest.mark.filterwarnings(
        "ignore:libgalpy_actionAngleTorus C extension module not loaded"
    ),
    pytest.mark.filterwarnings("ignore:Using C implementation to integrate orbits"),
]

RO, VO = 8.0, 220.0  # kpc, km/s: galpy's natural units

# agama's points (kpc, km/s) on the tori (J_r, J_z, J_phi) = (84, 12, 966),
# (25, 56, 842), (54, 58, 662), (174, 50, 1037) and (75, 30, 1697) kpc km/s at
# angles (0.3, 1.1, 0), and ten periods of each torus's slowest frequency (Myr).
POINTS = [
    (
        [3.366368325, 0.6347236355, 0.3750658095, 1.015833935, 287.147656, 18.9475702],
        1398.3,
    ),
    (
        [
            3.441623919,
            0.3079394179,
            0.9509377176,
            -1.507762661,
            244.517042,
            36.63456253,
        ],
        1278.1,
    ),
    (
        [
            2.570227034,
            0.4237811163,
            0.8805000304,
            -1.175146889,
            257.3710362,
            41.33428654,
        ],
        1162.0,
    ),
    (
        [
            3.384972941,
            1.105129641,
            0.9338240261,
            -3.274576743,
            305.2849125,
            16.12421689,
        ],
        1729.4,
    ),
    (
        [
            6.114954918,
            0.8622457933,
            0.8820334939,
            -4.540426873,
            276.8761273,
            17.36672661,
        ],
        2229.4,
    ),
]

# Measured worst relative error per action (galpy 1.11.1, default settings):
# 3.5e-2, the J_z = 12 of the first orbit. Most of it is galpy's: against the
# torus actions, galpy's J_r and J_z are off by 0.2 to 3.9 percent on these
# orbits and O2GF's by at most 0.9 percent. Over b from 0.15 to 0.5 (natural
# units) and integration times of 100 and 400, galpy's J_z of the first orbit
# stays 3.5 to 4.7 percent above the torus. The other orbits agree within
# 2.7e-2. The tolerance is 2x the worst.
RTOL = 7e-2


def _galpy_pot() -> list:
    from galpy import potential as gpp  # deferred: optional

    G = float(gp.MilkyWayPotential2022().constants["G"].ustrip("kpc km2 / (s2 Msun)"))

    def amp(m: float) -> float:
        return G * m / (RO * VO**2)

    out = []
    for kind, p in mw2022_parts():
        if kind == "MiyamotoNagai":
            out.append(
                gpp.MiyamotoNagaiPotential(
                    amp=amp(p["m"]), a=p["a"] / RO, b=p["b"] / RO
                )
            )
        elif kind == "NFW":
            out.append(gpp.NFWPotential(amp=amp(p["m"]), a=p["r_s"] / RO))
        else:  # galpy's Hernquist amp is twice the total mass
            out.append(gpp.HernquistPotential(amp=2 * amp(p["m"]), a=p["r_s"] / RO))
    return out


@pytest.mark.parametrize("point", POINTS, ids=[f"orbit{i}" for i in range(len(POINTS))])
def test_vs_galpy_isochroneapprox(point: tuple) -> None:
    from galpy.actionAngle import (  # deferred: optional
        actionAngleIsochroneApprox,
        estimateBIsochrone,
    )

    xv, tmax = np.asarray(point[0]), point[1]
    res, Jg = o2gf_from_point(gp.MilkyWayPotential2022(), xv, tmax)
    assert not bool(res.flagged)

    pots = _galpy_pot()
    R, phi = np.hypot(xv[0], xv[1]), np.arctan2(xv[1], xv[0])
    v_R = (xv[0] * xv[3] + xv[1] * xv[4]) / R
    v_T = (xv[0] * xv[4] - xv[1] * xv[3]) / R
    b = estimateBIsochrone(pots, R / RO, xv[2] / RO, quantity=False)
    aA = actionAngleIsochroneApprox(pot=pots, b=b)
    jr, lz, jz = aA(
        R / RO, v_R / VO, v_T / VO, xv[2] / RO, xv[5] / VO, phi, quantity=False
    )
    Jp = np.ravel([jr, lz, jz]) * RO * VO  # kpc km/s, (r, phi, z)

    np.testing.assert_allclose(Jg, Jp, rtol=RTOL)


# The isochrone orbit of tests/actionangle/test_o2gf.py, in kpc, km/s and Myr.
ISO_M, ISO_B = 1e11, 2.0  # Msun, kpc
ISO_XV = [6.0, 0.0, 1.0, 0.0, 127.11298, 48.889614]
ISO_TMAX = 1100.0
# Measured worst relative error per action (galpy 1.11.1): 4.2e-11, on J_r.
# With a fitted toy O2GF lands on the true isochrone and S_n is near 0, so
# this is the toy fit's tolerance, not a truncation error. The tolerance is 2x
# the worst, rounded up.
ISO_RTOL = 1e-10


def test_vs_galpy_isochrone() -> None:
    from galpy.actionAngle import actionAngleIsochrone  # deferred: optional
    from galpy.potential import IsochronePotential  # deferred: optional

    pot = gp.IsochronePotential(
        m_tot=u.Q(ISO_M, "Msun"), r_s=u.Q(ISO_B, "kpc"), units="galactic"
    )
    xv = np.asarray(ISO_XV)
    res, Jg = o2gf_from_point(pot, xv, ISO_TMAX)
    assert not bool(res.flagged)

    G = float(pot.constants["G"].ustrip("kpc km2 / (s2 Msun)"))
    aA = actionAngleIsochrone(
        ip=IsochronePotential(amp=G * ISO_M / (RO * VO**2), b=ISO_B / RO)
    )
    R, phi = np.hypot(xv[0], xv[1]), np.arctan2(xv[1], xv[0])
    v_R = (xv[0] * xv[3] + xv[1] * xv[4]) / R
    v_T = (xv[0] * xv[4] - xv[1] * xv[3]) / R
    jr, lz, jz = aA(
        R / RO, v_R / VO, v_T / VO, xv[2] / RO, xv[5] / VO, phi, quantity=False
    )
    Jp = np.ravel([jr, lz, jz]) * RO * VO  # kpc km/s, (r, phi, z)

    np.testing.assert_allclose(Jg, Jp, rtol=ISO_RTOL)

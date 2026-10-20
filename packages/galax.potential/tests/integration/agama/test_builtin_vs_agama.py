"""Closed-form builtin potentials against agama's implementations.

Every row is analytic on both sides. Potential, gradient, density and hessian
are each compared.
"""

from types import ModuleType
from typing import Any

import pytest

import unxt as u

import galax.potential as gp
from ...unit.potential.optional_deps import OptDeps
from ._compare import assert_matches_agama

pytestmark = [
    pytest.mark.agama,
    pytest.mark.skipif(not OptDeps.AGAMA.installed, reason="requires agama"),
]

T0 = u.Q(0, "Myr")

# Both sides are closed forms, so this would be ~1e-15 but for G: agama's
# (4.30091727068e-6 kpc km2 / s2 / Msun) exceeds galax's -- astropy's CODATA
# 2018 value -- by 1.49e-10 relative, which every G-scaled quantity carries.
RTOL = 2e-10


def _mn(p: gp.MiyamotoNagaiPotential) -> dict[str, Any]:
    return {
        "type": "MiyamotoNagai",
        "mass": p.m_tot(T0, ustrip="Msun"),
        "scaleRadius": p.a(T0, ustrip="kpc"),
        "scaleHeight": p.b(T0, ustrip="kpc"),
    }


def _nfw(p: gp.NFWPotential) -> dict[str, Any]:
    return {
        "type": "NFW",
        "mass": p.m(T0, ustrip="Msun"),
        "scaleRadius": p.r_s(T0, ustrip="kpc"),
    }


def _hernquist(p: gp.HernquistPotential) -> dict[str, Any]:
    return {
        "type": "Dehnen",
        "gamma": 1,
        "mass": p.m_tot(T0, ustrip="Msun"),
        "scaleRadius": p.r_s(T0, ustrip="kpc"),
    }


_MW = gp.MilkyWayPotential()
_MW2022 = gp.MilkyWayPotential2022()

CASES = [
    pytest.param(
        gp.PlummerPotential(m_tot=1e11, r_s=2.0, units="galactic"),
        {"type": "Plummer", "mass": 1e11, "scaleRadius": 2.0},
        id="plummer",
    ),
    pytest.param(
        gp.IsochronePotential(m_tot=1e11, r_s=5.0, units="galactic"),
        {"type": "Isochrone", "mass": 1e11, "scaleRadius": 5.0},
        id="isochrone",
    ),
    pytest.param(
        gp.NFWPotential(m=5.4e11, r_s=15.62, units="galactic"),
        {"type": "NFW", "mass": 5.4e11, "scaleRadius": 15.62},
        id="nfw",
    ),
    pytest.param(
        gp.MiyamotoNagaiPotential(m_tot=6.8e10, a=3.0, b=0.28, units="galactic"),
        {
            "type": "MiyamotoNagai",
            "mass": 6.8e10,
            "scaleRadius": 3.0,
            "scaleHeight": 0.28,
        },
        id="miyamotonagai",
    ),
    pytest.param(
        gp.HernquistPotential(m_tot=5e9, r_s=1.0, units="galactic"),
        {"type": "Dehnen", "gamma": 1, "mass": 5e9, "scaleRadius": 1.0},
        id="hernquist",
    ),
    pytest.param(
        gp.JaffePotential(m_tot=1e12, r_s=10.0, units="galactic"),
        {"type": "Dehnen", "gamma": 2, "mass": 1e12, "scaleRadius": 10.0},
        id="jaffe",
    ),
    pytest.param(
        gp.LogarithmicPotential(v_c=u.Q(220.0, "km / s"), r_s=8.0, units="galactic"),
        {"type": "Logarithmic", "v0": 220.0, "scaleRadius": 8.0},
        id="logarithmic",
    ),
    pytest.param(
        gp.LMJ09LogarithmicPotential(
            v_c=u.Q(220.0, "km / s"),
            r_s=12.0,
            q1=1.0,
            q2=0.9,
            q3=0.8,
            phi=u.Q(20, "deg"),
            units="galactic",
        ),
        {
            "type": "Logarithmic",
            "v0": 220.0,
            "scaleRadius": 12.0,
            "axisRatioY": 0.9,
            "axisRatioZ": 0.8,
            # agama's Euler angles are in radians; the first is galax's `phi`.
            "orientation": [u.ustrip("rad", u.Q(20, "deg")), 0.0, 0.0],
        },
        id="lmj09logarithmic",
    ),
    pytest.param(
        gp.HarmonicOscillatorPotential(omega=u.Q(0.05, "1 / Myr"), units="galactic"),
        {"type": "Harmonic", "Omega": u.ustrip("km / (s kpc)", u.Q(0.05, "1 / Myr"))},
        id="harmonic",
    ),
    pytest.param(
        gp.KeplerPotential(m_tot=1e11, units="galactic"),
        {"type": "Plummer", "mass": 1e11, "scaleRadius": 0.0},
        id="kepler",
    ),
    pytest.param(
        _MW,
        [_mn(_MW.disk), _nfw(_MW.halo), _hernquist(_MW.bulge), _hernquist(_MW.nucleus)],
        id="milkywaypotential",
    ),
    pytest.param(
        _MW2022,
        [
            *(_mn(mn) for mn in _MW2022.disk._get_mn_components(T0)),
            _nfw(_MW2022.halo),
            _hernquist(_MW2022.bulge),
            _hernquist(_MW2022.nucleus),
        ],
        id="milkywaypotential2022",
    ),
]


@pytest.mark.parametrize(("pot", "spec"), CASES)
def test_matches_agama(
    agama: ModuleType,
    pot: gp.AbstractPotential,
    spec: dict | list[dict],
) -> None:
    """Potential, gradient, density and hessian agree with agama."""
    components = spec if isinstance(spec, list) else [spec]
    assert_matches_agama(pot, agama.Potential(*components), rtol=RTOL)

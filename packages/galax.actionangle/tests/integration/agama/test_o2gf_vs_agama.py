"""O2GF actions against agama's torus mapper and its Staeckel fudge.

Both methods are independent of O2GF, so the actions agree only to within the
errors of the two methods, and the tolerances come from measuring them. Each
orbit is integrated by galax for ten periods of its slowest frequency, sampled
2000 times, and run through O2GF at n_max = 8.

agama orders every triplet ``(r, z, phi)``; galax orders them ``(r, phi, z)``.
"""

from types import ModuleType

import numpy as np
import pytest

import unxt as u

import galax.potential as gp
from ...actionangle._helpers import mw2022_parts, o2gf_from_point
from ...actionangle.optional_deps import OptDeps

pytestmark = [
    pytest.mark.agama,
    pytest.mark.skipif(not OptDeps.AGAMA.installed, reason="requires agama"),
]

_GALAX_ORDER = [0, 2, 1]  # agama's (r, z, phi) -> galax's (r, phi, z)
_PER_MYR = float(u.ustrip("1/Myr", u.Q(1.0, "km / (s kpc)")))

# Tori (J_r, J_z, J_phi) in kpc km/s, in MilkyWayPotential2022, each mapped at
# the angles ANGLES. None is planar. With n_max = 8, ten periods are too short
# for most disk tori: some low-order n . Omega is slow enough that its angle
# covers less than 2 pi, and O2GF flags the orbit. These tori were picked from
# a random sample as ones whose modes are all covered. The near-resonant torus
# NEAR_RESONANT is tested on its own below.
TORI = [
    (84.0, 12.0, 966.0),
    (25.0, 56.0, 842.0),
    (54.0, 58.0, 662.0),
    (174.0, 50.0, 1037.0),
    (75.0, 30.0, 1697.0),
]
ANGLES = (0.3, 1.1, 0.0)
TORUS_TOL = 1e-3  # agama's torus accuracy. 3e-4 and below fail on most tori.

# Measured worst relative error of each action (agama 1.0.161) over the five
# tori, in galax's order: J_r 8.7e-3, the J_r = 25 of the near-circular
# (25, 56, 842); J_phi 8.4e-5, the same torus; J_z 2.9e-3, the J_z = 12 of
# (84, 12, 966). These are O2GF's truncation error at n_max = 8 together with
# the torus's own error at TORUS_TOL: the J_r of (25, 56, 842) moves from 2.6e-3
# to 8.7e-3 between n_max = 6 and 8, and the J_r of (84, 12, 966) drops from
# 9.4e-4 to 8.8e-5 when the torus is built at 1e-4. Each tolerance is 2x its
# worst, rounded up.
TORUS_RTOL = np.array([1.8e-2, 1.7e-4, 5.9e-3])  # (J_r, J_phi, J_z)

# Near the 3:2 Omega_z : Omega_r resonance O2GF can be wrong without flagging
# the orbit. On this torus and on (186, 12, 697) O2GF is unflagged (no mode is
# badly covered), yet its J_z is off from the torus by 10 to 22 percent. The
# torus and agama's Staeckel fudge agree with each other to about 4 percent,
# so O2GF is the outlier. Here its J_z is 12.5, against 16 from the torus and
# 15.3 from the fudge. A longer orbit does not fix it: at 20 periods its J_z is
# still 10 percent low, and that of (186, 12, 697) is 15 percent high. A
# near-resonance flag in O2GF is a follow-up.
NEAR_RESONANT = (135.0, 16.0, 645.0)

# Miyamoto-Nagai disk, and orbits as (x, y, z) kpc and (v_R, v_phi / v_c(R),
# v_z) km/s at y = 0. Both are unflagged at n_max = 8.
MN = {"m": 1e11, "a": 3.0, "b": 0.3}
MN_ORBITS = {
    "eccentric": ([10.0, 0.0, 0.3], [25.0, 1.0, 30.0]),
    "inner": ([4.0, 0.0, 0.1], [40.0, 0.85, 25.0]),
}
# Measured worst relative error per action (agama 1.0.161): 2.5e-3, the
# J_z = 2.6 of "inner"; "eccentric" agrees within 8.0e-4. The Staeckel fudge is
# not exact for a Miyamoto-Nagai disk: mapping the fudge's actions and angles of
# "inner" through agama's torus and back into the fudge moves its J_z by
# 4.5e-3. The tolerance is 2x the worst.
STAECKEL_RTOL = 5e-3


def _agama_part(kind: str, p: dict[str, float]) -> dict:
    if kind == "MiyamotoNagai":
        return {
            "type": "MiyamotoNagai",
            "mass": p["m"],
            "scaleRadius": p["a"],
            "scaleHeight": p["b"],
        }
    if kind == "NFW":
        return {"type": "NFW", "mass": p["m"], "scaleRadius": p["r_s"]}
    # Hernquist is Dehnen with gamma = 1.
    return {"type": "Dehnen", "gamma": 1, "mass": p["m"], "scaleRadius": p["r_s"]}


def _tmax(om: np.ndarray) -> float:
    """Ten periods of the slowest frequency, in Myr."""
    return float(10 * 2 * np.pi / (np.min(om) * _PER_MYR))


@pytest.fixture(scope="module")
def mapper(agama: ModuleType):
    pot = agama.Potential(*[_agama_part(k, p) for k, p in mw2022_parts()])
    return agama.ActionMapper(pot, tol=TORUS_TOL)


@pytest.mark.parametrize("J", TORI, ids=[str(J) for J in TORI])
def test_torus_round_trip(mapper, J: tuple) -> None:
    xv, om = mapper(np.r_[J, ANGLES], frequencies=True)
    res, Jg = o2gf_from_point(gp.MilkyWayPotential2022(), xv, _tmax(om))
    assert not bool(res.flagged)
    ref = np.asarray(J)[_GALAX_ORDER]
    for i, name in enumerate(("J_r", "J_phi", "J_z")):
        np.testing.assert_allclose(Jg[i], ref[i], rtol=TORUS_RTOL[i], err_msg=name)


@pytest.fixture(scope="module")
def near_resonant(mapper) -> tuple:
    xv, om = mapper(np.r_[NEAR_RESONANT, ANGLES], frequencies=True)
    return o2gf_from_point(gp.MilkyWayPotential2022(), xv, _tmax(om))


def test_near_resonant_torus_unflagged(near_resonant) -> None:
    """O2GF does not flag the near-resonant torus, though its J_z is wrong."""
    res, _ = near_resonant
    assert not bool(res.flagged)
    assert int(res.n_bad_cover) == 0


@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="O2GF's J_z is 22 percent low near the 3:2 resonance, unflagged",
)
def test_near_resonant_torus_j_z(near_resonant) -> None:
    _, Jg = near_resonant
    np.testing.assert_allclose(Jg[2], NEAR_RESONANT[1], rtol=TORUS_RTOL[2])


@pytest.mark.parametrize("name", MN_ORBITS)
def test_vs_agama_staeckel(agama: ModuleType, name: str) -> None:
    apot = agama.Potential(
        type="MiyamotoNagai", mass=MN["m"], scaleRadius=MN["a"], scaleHeight=MN["b"]
    )
    x, (v_r, f_c, v_z) = MN_ORBITS[name]
    v_c = np.sqrt(-x[0] * apot.force([x[0], 0.0, 0.0])[0])
    xv = np.array([*x, v_r, f_c * v_c, v_z])
    aJ, _, aO = agama.ActionFinder(apot)(xv, angles=True)

    pot = gp.MiyamotoNagaiPotential(
        m_tot=u.Q(MN["m"], "Msun"),
        a=u.Q(MN["a"], "kpc"),
        b=u.Q(MN["b"], "kpc"),
        units="galactic",
    )
    res, Jg = o2gf_from_point(pot, xv, _tmax(aO))
    assert not bool(res.flagged)
    np.testing.assert_allclose(Jg, aJ[_GALAX_ORDER], rtol=STAECKEL_RTOL)

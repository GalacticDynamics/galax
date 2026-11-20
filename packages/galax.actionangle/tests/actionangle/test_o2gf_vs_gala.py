"""Compare O2GF with gala's ``find_actions_o2gf`` on stored reference orbits.

The orbits and gala's outputs are in ``data/o2gf_gala_refs.npz``, made by
``data/make_o2gf_gala_refs.py`` with gala 1.12.0 (commit bebac7d72). gala is not
needed to run these tests. Each orbit was integrated by gala, so both sides
see the same samples. gala's G and galax's G in "galactic" units are the same
float64 (both come from astropy's CODATA 2022 value), so the isochrone toy
mass needs no conversion.
"""

import pathlib

import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.actionangle import (
    IsochroneToy,
    OscillatorToy,
    o2gf_actions_angles_freqs as o2gf,
)

DATA = np.load(pathlib.Path(__file__).parent / "data" / "o2gf_gala_refs.npz")
CASES = tuple(str(c) for c in DATA["cases"])

# The galax potential for each case. With a fixed toy it only supplies the
# units and G. Without one it seeds the isochrone fit. galax has no NFW with
# gala's flattened potential (a, b, c), so the NFW cases seed from the
# spherical NFW with the same m and r_s.
_NFW = gp.NFWPotential(m=u.Q(1e12, "Msun"), r_s=u.Q(20.0, "kpc"), units="galactic")
POTS = {
    "mw2022_loop": gp.MilkyWayPotential2022(),
    "flat_nfw_loop": _NFW,
    "mn_loop": gp.MiyamotoNagaiPotential(
        m_tot=u.Q(1e11, "Msun"), a=u.Q(3.0, "kpc"), b=u.Q(0.3, "kpc"), units="galactic"
    ),
    "triaxial_nfw_xtube": _NFW,
    "log_box": gp.LMJ09LogarithmicPotential(
        v_c=u.Q(200.0, "km/s"),
        r_s=u.Q(3.0, "kpc"),
        q1=1.0,
        q2=0.9,
        q3=0.8,
        phi=u.Q(0.0, "rad"),
        units="galactic",
    ),
}
G = float(POTS["mw2022_loop"].constants["G"].ustrip("kpc3 / (solMass Myr2)"))

# Tolerances (rtol on actions, rtol on freqs, atol on angles) for the own-fit
# comparison. gala fits the toy with Powell and galax with Levenberg-Marquardt.
# Powell stops early in the flat valley of (m, b), and galax's toy always has
# the lower energy scatter. The O2GF results move with the toy by the
# truncation error of the generating function at this n_max. Measured
# differences (actions, freqs, angles in rad):
#   mw2022_loop          3.0e-5   6.7e-9   6.8e-6
#   flat_nfw_loop        4.7e-11  1.6e-12  1.1e-10
#   mn_loop              3.4e-4   1.5e-7   2.9e-5
#   triaxial_nfw_xtube   3.4e-2   7.8e-2   9.0e-1
#   log_box              2.9e-6   5.9e-9   6.4e-7
# Each tolerance is 1e-4 or about 3 times the measured value, whichever is
# larger. The long-axis tube is very sensitive to the toy. In gala alone,
# scaling the fitted m by 0.8 or 1.25 (and b by the square root) moves its
# actions by up to 4%, its freqs by 8% and its angles by 2.9 rad, so its
# angles are only checked to 2 rad.
OWN_FIT_TOL = {
    "mw2022_loop": (1e-4, 1e-4, 1e-4),
    "flat_nfw_loop": (1e-4, 1e-4, 1e-4),
    "mn_loop": (1.1e-3, 1e-4, 1e-4),
    "triaxial_nfw_xtube": (0.1, 0.25, 2.0),
    "log_box": (1e-4, 1e-4, 1e-4),
}


def _case(name):
    pre = f"{name}__"
    return {k.removeprefix(pre): DATA[k] for k in DATA.files if k.startswith(pre)}


def _toy(c):
    if "toy_m" in c:
        return IsochroneToy(m=jnp.asarray(c["toy_m"]), b=jnp.asarray(c["toy_b"]))
    return OscillatorToy(omega=jnp.asarray(c["toy_omega"]))


def _scatter(toy, q, p):
    """Return gala's fit objective, the sum of (H - mean H)^2 in the toy."""
    T = 0.5 * np.sum(p**2, axis=-1)
    if isinstance(toy, IsochroneToy):
        m, b = float(toy.m), float(toy.b)
        H = T - G * m / (b + np.sqrt(b**2 + np.sum(q**2, axis=-1)))
    else:
        H = T + 0.5 * np.sum((np.asarray(toy.omega) * q) ** 2, axis=-1)
    return np.sum((H - H.mean()) ** 2)


def _angle_diff(a, b):
    """Difference of two angles, wrapped into [-pi, pi)."""
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


def _expected_axis(circ):
    """Return the axis swapped with z, by gala's rule: z, then x before y."""
    return 2 if circ[2] or not circ.any() else int(np.argmax(circ[:2]))


def test_reference_data():
    assert str(DATA["gala_version"]) == "1.12.0"
    assert set(CASES) == set(POTS)
    # One box orbit (no circulation) and one tube about the long axis x.
    np.testing.assert_array_equal(_case("log_box")["circulation"], [0, 0, 0])
    np.testing.assert_array_equal(_case("triaxial_nfw_xtube")["circulation"], [1, 0, 0])


@pytest.mark.parametrize("name", CASES)
def test_matches_gala_fixed_toy(name):
    c = _case(name)
    res = o2gf(POTS[name], c["q"], c["p"], c["t"], n_max=int(c["n_max"]), toy=_toy(c))
    np.testing.assert_array_equal(res.nvecs, c["nvecs"])
    assert bool(res.is_loop) == bool(c["circulation"].any())
    assert int(res.aligned_axis) == _expected_axis(c["circulation"])
    assert not bool(res.flagged)
    assert int(res.n_resonant) == 0

    np.testing.assert_allclose(res.aaf.actions, c["actions"], rtol=1e-10)
    np.testing.assert_allclose(res.aaf.freqs, c["freqs"], rtol=1e-10)
    np.testing.assert_allclose(_angle_diff(res.aaf.angles, c["angles"]), 0, atol=1e-10)
    # S_n holds entries near 0, so rtol is taken relative to the largest |S_n|.
    np.testing.assert_allclose(
        res.sn, c["sn"], rtol=1e-10, atol=1e-10 * np.max(np.abs(c["sn"]))
    )
    # dS_n/dJ comes from the larger, worse-conditioned angle system. The
    # largest measured difference is 4.7e-10 of max |dS_n/dJ| (the long-axis
    # tube).
    np.testing.assert_allclose(
        res.dsn_dj, c["dsn_dj"], rtol=0, atol=2e-9 * np.max(np.abs(c["dsn_dj"]))
    )


@pytest.mark.parametrize("name", CASES)
def test_matches_gala_own_fit(name):
    c = _case(name)
    res = o2gf(POTS[name], c["q"], c["p"], c["t"], n_max=int(c["n_max"]))
    is_loop = bool(c["circulation"].any())
    assert bool(res.is_loop) == is_loop
    assert bool(res.fit_converged)
    assert not bool(res.flagged)

    # galax's toy fits the orbit at least as well as gala's.
    toy = res.toy[0] if is_loop else res.toy[1]
    assert _scatter(toy, c["q"], c["p"]) <= _scatter(_toy(c), c["q"], c["p"]) * (
        1 + 1e-9
    )

    # S_n is relative to the toy, so it is not compared across toys.
    rtol_j, rtol_om, atol_th = OWN_FIT_TOL[name]
    np.testing.assert_allclose(res.aaf.actions, c["actions"], rtol=rtol_j)
    np.testing.assert_allclose(res.aaf.freqs, c["freqs"], rtol=rtol_om)
    np.testing.assert_allclose(
        _angle_diff(res.aaf.angles, c["angles"]), 0, atol=atol_th
    )

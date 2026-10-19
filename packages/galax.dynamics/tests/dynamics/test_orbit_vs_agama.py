"""Orbits integrated by galax against agama's, in the same potentials.

The potentials are closed forms that the `galax.potential` agama tests show
agree, so a disagreement here is the integrators'. Both are run tighter than
the tolerances below, which then pin the regime rather than the digits.
"""

from types import ModuleType

import diffrax as dfx
import jax
import numpy as np
import pytest
from plum import convert

import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from .optional_deps import OptDeps

pytestmark = [
    pytest.mark.agama,
    pytest.mark.skipif(not OptDeps.AGAMA.installed, reason="requires agama"),
]

# agama's unit of time with length in kpc and velocity in km/s.
KPC_PER_KMS = u.Q(1, "kpc") / u.Q(1, "km/s")

INTEGRATOR = gd.integrate.Integrator(
    gd.OrbitSolver(
        solver=dfx.Dopri8(scan_kind="bounded"),
        stepsize_controller=dfx.PIDController(rtol=1e-13, atol=1e-13),
        max_steps=2**18,
    )
)


@pytest.fixture(scope="module")
def agama() -> ModuleType:
    """Import agama and set its units to Msun / kpc / km/s -- *not* G = 1."""
    import agama  # deferred: optional, and built by hand

    assert jax.config.jax_enable_x64, "agama comparisons need JAX_ENABLE_X64=True"
    agama.setUnits(mass=1, length=1, velocity=1)
    return agama


@pytest.mark.parametrize(
    ("pot", "spec", "q0", "p0"),
    [
        pytest.param(
            gp.MiyamotoNagaiPotential(m_tot=6.8e10, a=3.0, b=0.28, units="galactic"),
            {
                "type": "MiyamotoNagai",
                "mass": 6.8e10,
                "scaleRadius": 3.0,
                "scaleHeight": 0.28,
            },
            [8.0, 0.0, 0.1],
            [0.0, 190.0, 5.0],
            id="disk-near-circular",
        ),
        pytest.param(
            gp.NFWPotential(m=5.4e11, r_s=15.62, units="galactic"),
            {"type": "NFW", "mass": 5.4e11, "scaleRadius": 15.62},
            [30.0, 0.0, 5.0],
            [0.0, 60.0, 30.0],
            id="halo-eccentric",
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
            [10.0, 2.0, 3.0],
            [20.0, 150.0, 40.0],
            id="triaxial",
        ),
    ],
)
def test_orbit_matches_agama(
    agama: ModuleType,
    pot: gp.AbstractPotential,
    spec: dict,
    q0: list[float],
    p0: list[float],
) -> None:
    """Positions and velocities over 20 periods agree, and both conserve energy.

    Measured (agama 1.0.0, ``accuracy=1e-14``; galax Dopri8 at
    ``rtol=atol=1e-13``), worst over 201 samples, relative to ``|q|``/``|v|``:
    disk 5.1e-12 / 5.2e-12, halo 2.3e-11 / 1.7e-11, triaxial 1.1e-11 /
    1.1e-11. Energy drift: galax 7e-13 to 1.4e-12, agama 5e-15 to 6e-14.
    """
    # agama's G exceeds galax's (astropy's) by 1.49e-10, which over 20 periods
    # shifts the phase by ~1e-8. Give agama the same G*M instead.
    if "mass" in spec:
        g_galax = u.ustrip("kpc km2 / (s2 Msun)", pot.constants["G"])
        spec = spec | {"mass": spec["mass"] * float(g_galax) / agama.G}
    agama_pot = agama.Potential(**spec)

    ic = np.array(q0 + p0)
    period = float(agama_pot.Tcirc(ic))
    t_agama, traj = agama.orbit(
        potential=agama_pot,
        ic=ic,
        time=20 * period,
        trajsize=201,
        accuracy=1e-14,
        dtype="float64",  # the default float32 caps agreement at ~5e-8
    )

    ts = u.Q(t_agama * u.ustrip("Myr", KPC_PER_KMS), "Myr")
    w0 = gc.PhaseSpaceCoordinate(q=u.Q(q0, "kpc"), p=u.Q(p0, "km/s"), t=ts[0])
    orbit = gd.evaluate_orbit(pot, w0, ts, integrator=INTEGRATOR)

    q = np.asarray(u.ustrip("kpc", convert(orbit.q, u.Q)))
    v = np.asarray(u.ustrip("km/s", convert(orbit.p, u.Q)))
    for name, got, want in (("q", q, traj[:, :3]), ("v", v, traj[:, 3:])):
        err = np.linalg.norm(got - want, axis=-1) / np.linalg.norm(want, axis=-1)
        assert np.max(err) < 1e-9, f"{name}: {np.max(err):.2e}"

    energy = np.asarray(u.ustrip("km2/s2", gp.total_energy(pot, orbit)))
    agama_energy = agama_pot.potential(traj[:, :3]) + 0.5 * np.sum(
        traj[:, 3:] ** 2, axis=-1
    )
    np.testing.assert_allclose(energy, energy[0], rtol=1e-11, err_msg="galax drift")
    np.testing.assert_allclose(
        agama_energy, agama_energy[0], rtol=1e-11, err_msg="agama drift"
    )

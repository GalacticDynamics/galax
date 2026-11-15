"""`make_mw_lmc_potential` against a joint MW + LMC + star integration."""

import diffrax as dfx
import pytest

import diffraxtra as dfxtra
import quaxed.numpy as jnp
import unxt as u

import galax.dynamics as gd
import galax.potential as gp

MW = gp.HernquistPotential(m_tot=1e12, r_s=20, units="galactic")
LMC = gp.HernquistPotential(m_tot=1.5e11, r_s=15, units="galactic")
MW_W0 = (jnp.zeros(3), jnp.zeros(3))  # [kpc], [kpc/Myr]
LMC_W0 = (
    jnp.array([-0.8, -41.5, -26.8]),
    u.ustrip("kpc/Myr", u.Q([-56.0, -219, 186], "km/s")),
)
Q0 = jnp.array([[10.0, 0, 0], [0, 20, 5], [-30, 0, 40]])
P0 = u.ustrip("kpc/Myr", u.Q([[0.0, 200, 0], [-150, 0, 50], [0, 120, 30]], "km/s"))
T1 = -500.0  # [Myr]
TIGHT = {"stepsize_controller": dfx.PIDController(rtol=1e-12, atol=1e-12)}
SOLVER = dfxtra.DiffEqSolver(dfx.Dopri8(), max_steps=None, **TIGHT)


def joint_orbit(q0, p0):
    """Integrate the MW centre, LMC and stars together in the inertial frame."""

    def vf(t, y, _):
        (xm, vm), (xl, vl), (xs, vs) = y
        acc = lambda pot, x: -pot._gradient(x, t)
        as_ = acc(MW, xs - xm) + acc(LMC, xs - xl)
        return ((vm, acc(LMC, xm - xl)), (vl, acc(MW, xl - xm)), (vs, as_))

    y0 = (MW_W0, LMC_W0, (q0, p0))
    soln = dfx.diffeqsolve(
        dfx.ODETerm(vf),
        dfx.Dopri8(),
        0.0,
        T1,
        -1.0,
        y0,
        max_steps=None,
        **TIGHT,
    )
    return soln.ys[2][0][0]


def mw_lmc_orbit(q0, p0, pot):
    soln = gd.OrbitSolver(**TIGHT).solve(
        gd.fields.HamiltonianField(pot), (q0, p0), 0.0, T1, dt0=-1.0
    )
    return soln.ys[0][0]


def make_pot():
    # Friction is switched off: `b_coulomb_min` beyond the LMC's distance
    # makes the Coulomb logarithm zero.
    return gd.examples.make_mw_lmc_potential(
        MW,
        LMC,
        MW_W0,
        LMC_W0,
        b_coulomb_min=u.Q(1e6, "kpc"),
        t_interp=u.Q(jnp.linspace(0, T1, 2001), "Myr"),
        solver_kwargs={"solver": SOLVER},
    )


@pytest.mark.parametrize("n", [None, 1, 3], ids=["unbatched", "batch1", "batch3"])
def test_matches_joint_integration(n):
    """Stars in the composite follow the joint inertial integration."""
    q0, p0 = (Q0[0], P0[0]) if n is None else (Q0[:n], P0[:n])
    got = mw_lmc_orbit(q0, p0, make_pot())
    want = joint_orbit(q0, p0)
    assert got.shape == want.shape
    assert jnp.max(jnp.abs(got - want)) < 1e-6

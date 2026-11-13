"""Tests for the variational equations of orbit fields."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import unxt as u

import galax.dynamics as gd
import galax.potential as gp
from ._variational_helpers import LinearDragField, make_w0
from galax.dynamics._src.orbit.variational import compute_variational, tangent_terms

POT = gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
FIELD = gd.fields.HamiltonianField(POT)
X = jnp.array([8.0, 0, 1])
V = jnp.array([0.0, 0.2, 0.02])
EYE = jnp.eye(6)


# ===================================================================
# tangent_terms


def test_tangent_odeterm_matches_jacobian():
    """The augmented ODE term returns the field and its Jacobian times dW."""
    term = FIELD.terms(dfx.Dopri8())
    aug = tangent_terms(term)
    (xd, dxd), (vd, dvd) = aug.vf(0.0, ((X, EYE[:3]), (V, EYE[3:])), None)

    def rhs(w):
        return jnp.concat(term.vf(0.0, (w[:3], w[3:]), None))

    w = jnp.concat([X, V])
    jac = jax.jacfwd(rhs)(w)
    assert jnp.allclose(jnp.concat([xd, vd]), rhs(w))
    assert jnp.allclose(jnp.concat([dxd, dvd]), jac, rtol=1e-12)


@pytest.mark.parametrize(
    "solver", [gd.experimental.Leapfrog(), dfx.SemiImplicitEuler()]
)
def test_tangent_split_terms_match_jacobian(solver):
    """The augmented drift and kick terms return their Jacobians times dW."""
    f, g = FIELD.terms(solver)
    f_aug, g_aug = tangent_terms((f, g))

    vd, dvd = f_aug.vf(0.0, (V, EYE[3:]), None)
    assert jnp.array_equal(vd, f.vf(0.0, V, None))
    assert jnp.array_equal(dvd, EYE[3:])

    ad, dad = g_aug.vf(0.0, (X, EYE[:3]), None)
    jac = jax.jacfwd(lambda q: g.vf(0.0, q, None))(X)
    assert jnp.allclose(ad, g.vf(0.0, X, None))
    assert jnp.allclose(dad, jac @ EYE[:3], rtol=1e-12)


# ===================================================================
# compute_variational

TIGHT = gd.OrbitSolver(stepsize_controller=dfx.PIDController(rtol=1e-11, atol=1e-11))
XV0 = jnp.array([8.0, 0, 1, 0, 0.2, 0.02])  # kpc, kpc/Myr
TS = jnp.linspace(0.0, 500.0, 5)  # Myr
OMEGA = jnp.block([[jnp.zeros((3, 3)), jnp.eye(3)], [-jnp.eye(3), jnp.zeros((3, 3))]])


_w0 = make_w0


def _end_state(pot, xv, ts, t=0.0, solver=TIGHT):
    """Return the final (x, v) of `compute_orbit` as a flat galactic-unit array."""
    orbit = gd.orbit.compute_orbit(pot, _w0(xv, t), u.Q(ts, "Myr"), solver=solver)
    q = jnp.stack([u.ustrip("kpc", c) for c in orbit.q[-1].asdict().values()])
    p = jnp.stack([u.ustrip("kpc/Myr", c) for c in orbit.p[-1].asdict().values()])
    return jnp.concat([q, p])


def test_stm_is_identity_at_start():
    _, D = compute_variational(POT, _w0(XV0), u.Q(TS, "Myr"))
    assert D.shape == (5, 6, 6)
    assert jnp.allclose(D[0], jnp.eye(6), atol=1e-14)


def test_stm_harmonic_oscillator_analytic():
    w = 0.1
    pot = gp.HarmonicOscillatorPotential(omega=u.Q(w, "1/Myr"), units="galactic")
    ts = jnp.linspace(0.0, 100.0, 11)
    _, D = compute_variational(pot, _w0(XV0), u.Q(ts, "Myr"), solver=TIGHT)
    c, s = jnp.cos(w * ts)[:, None, None], jnp.sin(w * ts)[:, None, None]
    eye = jnp.eye(3)
    M = jnp.concat(
        [
            jnp.concat([c * eye, s / w * eye], axis=-1),
            jnp.concat([-w * s * eye, c * eye], axis=-1),
        ],
        axis=-2,
    )
    assert jnp.allclose(D, M, atol=1e-8)


def test_stm_matches_jacfwd_of_compute_orbit():
    _, D = compute_variational(POT, _w0(XV0), u.Q(TS, "Myr"), solver=TIGHT)
    jac = jax.jacfwd(lambda xv: _end_state(POT, xv, TS))(XV0)
    assert jnp.allclose(D[-1], jac, rtol=1e-6, atol=1e-9)


def test_stm_matches_central_differences():
    _, D = compute_variational(POT, _w0(XV0), u.Q(TS, "Myr"), solver=TIGHT)
    h = 1e-5
    cols = [
        (_end_state(POT, XV0 + h * e, TS) - _end_state(POT, XV0 - h * e, TS)) / (2 * h)
        for e in jnp.eye(6)
    ]
    # The O(h^2) truncation error of the central difference sets the tolerance.
    assert jnp.allclose(D[-1], jnp.stack(cols, axis=-1), rtol=1e-5, atol=1e-7)


def test_stm_is_symplectic():
    pot = gp.MilkyWayPotential()
    _, D = compute_variational(pot, _w0(XV0), u.Q(TS, "Myr"), solver=TIGHT)
    for M in D:
        assert jnp.allclose(M.T @ OMEGA @ M, OMEGA, atol=1e-8)
        assert jnp.allclose(jnp.linalg.det(M), 1.0, atol=1e-8)


def test_deviation_columns_are_stm_times_dw0():
    dw0 = jr.normal(jr.key(0), (6, 2))
    _, M = compute_variational(POT, _w0(XV0), u.Q(TS, "Myr"), solver=TIGHT)
    _, D = compute_variational(POT, _w0(XV0), u.Q(TS, "Myr"), dw0=dw0, solver=TIGHT)
    assert D.shape == (5, 6, 2)
    assert jnp.allclose(D, M @ dw0, rtol=1e-8, atol=1e-10)


def test_orbit_matches_compute_orbit():
    orbit, _ = compute_variational(POT, _w0(XV0), u.Q(TS, "Myr"), solver=TIGHT)
    ref = gd.orbit.compute_orbit(POT, _w0(XV0), u.Q(TS, "Myr"), solver=TIGHT)
    assert jnp.allclose(
        u.ustrip("kpc", orbit.q.x), u.ustrip("kpc", ref.q.x), rtol=1e-8, atol=1e-10
    )
    assert jnp.allclose(
        u.ustrip("kpc", orbit.q.z), u.ustrip("kpc", ref.q.z), rtol=1e-8, atol=1e-10
    )
    assert jnp.allclose(u.ustrip("Myr", orbit.t), TS)


@pytest.mark.parametrize(
    "method", [gd.experimental.Leapfrog(), dfx.SemiImplicitEuler()]
)
def test_symplectic_tangent_map_is_exact(method):
    solver = gd.OrbitSolver(method, stepsize_controller=dfx.ConstantStepSize())
    _, D = compute_variational(POT, _w0(XV0), u.Q(TS, "Myr"), solver=solver, dt0=0.5)

    terms = FIELD.terms(method)

    def end(xv):
        soln = dfx.diffeqsolve(
            terms,
            method,
            0.0,
            500.0,
            0.5,
            (xv[:3], xv[3:]),
            max_steps=2_000,
            adjoint=dfx.ForwardMode(),
        )
        return jnp.concat([soln.ys[0][-1], soln.ys[1][-1]])

    assert jnp.allclose(D[-1], jax.jacfwd(end)(XV0), rtol=1e-10, atol=1e-12)
    assert jnp.allclose(jnp.linalg.det(D[-1]), 1.0, atol=1e-12)


def test_velocity_dependent_field_liouville():
    gamma = 0.01
    field = LinearDragField(POT, gamma)
    _, D = compute_variational(field, _w0(XV0), u.Q(TS, "Myr"), solver=TIGHT)
    assert jnp.allclose(jnp.linalg.det(D), jnp.exp(-3 * gamma * TS), rtol=1e-8)


# -------------------------------------------------------------------
# Review focus


def test_backward_integration_inverts_forward():
    _, Df = compute_variational(POT, _w0(XV0), u.Q(TS, "Myr"), solver=TIGHT)
    xv1 = _end_state(POT, XV0, TS)
    _, Db = compute_variational(
        POT, _w0(xv1, t=500.0), u.Q(TS[::-1], "Myr"), solver=TIGHT
    )
    # M has entries of order 100, so the product carries ~1e-7 absolute error.
    assert jnp.allclose(Db[-1] @ Df[-1], jnp.eye(6), atol=1e-6)


def test_w0_time_before_ts0():
    _, D = compute_variational(POT, _w0(XV0, t=-100.0), u.Q(TS, "Myr"), solver=TIGHT)
    jac = jax.jacfwd(lambda xv: _end_state(POT, xv, TS, t=-100.0))(XV0)
    assert jnp.allclose(D[-1], jac, rtol=1e-6, atol=1e-9)


def test_single_output_time():
    _, D1 = compute_variational(POT, _w0(XV0), u.Q([500.0], "Myr"), solver=TIGHT)
    _, D = compute_variational(POT, _w0(XV0), u.Q(TS, "Myr"), solver=TIGHT)
    assert D1.shape == (1, 6, 6)
    assert jnp.allclose(D1[0], D[-1], rtol=1e-6, atol=1e-9)


def test_ts_units_are_converted():
    _, D_myr = compute_variational(POT, _w0(XV0), u.Q(TS, "Myr"))
    _, D_gyr = compute_variational(POT, _w0(XV0), u.Q(TS / 1e3, "Gyr"))
    assert jnp.allclose(D_myr, D_gyr, rtol=1e-12)


def test_batched_and_jitted():
    xv = jnp.stack([XV0, XV0 * jnp.array([1.1, 1, 1, 1, 0.9, 1])])

    @jax.jit
    def stm(xv):
        return compute_variational(POT, _w0(xv), u.Q(TS, "Myr"))[1]

    D = stm(xv)
    assert D.shape == (2, 5, 6, 6)
    assert jnp.allclose(D[0], stm(xv[0]), rtol=1e-12)
    assert jnp.allclose(D[1], stm(xv[1]), rtol=1e-12)


LEAPFROG = gd.OrbitSolver(
    gd.experimental.Leapfrog(), stepsize_controller=dfx.ConstantStepSize()
)


def test_constant_step_backward_integration():
    """A positive dt0 also works when integrating backward."""
    _, Df = compute_variational(POT, _w0(XV0), u.Q(TS, "Myr"), solver=LEAPFROG, dt0=0.5)
    orbit, _ = compute_variational(
        POT, _w0(XV0), u.Q(TS, "Myr"), solver=LEAPFROG, dt0=0.5
    )
    xv1 = jnp.concat(
        [
            jnp.stack([u.ustrip("kpc", c) for c in orbit.q[-1].asdict().values()]),
            jnp.stack([u.ustrip("kpc/Myr", c) for c in orbit.p[-1].asdict().values()]),
        ]
    )
    _, Db = compute_variational(
        POT, _w0(xv1, t=500.0), u.Q(TS[::-1], "Myr"), solver=LEAPFROG, dt0=0.5
    )
    assert jnp.allclose(Db[-1] @ Df[-1], jnp.eye(6), atol=1e-6)


def test_constant_step_mixed_directions():
    """Stage one runs forward and stage two backward; both use |dt0|."""
    _, D = compute_variational(
        POT, _w0(XV0), u.Q([500.0, 250.0], "Myr"), solver=LEAPFROG, dt0=0.5
    )
    assert jnp.all(jnp.isfinite(D))
    assert jnp.allclose(jnp.linalg.det(D), 1.0, atol=1e-10)

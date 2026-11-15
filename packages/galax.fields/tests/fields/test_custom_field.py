"""Tests for `galax.fields.CustomField`."""

# ruff: noqa: ARG005

from typing import Any

import diffrax as dfx

import diffraxtra as dfxtra
import quaxed.numpy as jnp
import unxt as u

import galax.fields as gf

K = 0.3
# the package default solver forces dtmin=0.05, too coarse for rtol=1e-6 here
SOLVER = dfxtra.DiffEqSolver(
    dfx.Dopri8(), stepsize_controller=dfx.PIDController(rtol=1e-10, atol=1e-10)
)


def decay(t: Any, y: Any, args: Any) -> Any:
    return -K * y


def test_call_returns_func_output() -> None:
    """Calling the field calls the function."""
    field = gf.CustomField(lambda t, y, args: -0.5 * y)
    assert jnp.allclose(field(0.0, jnp.array([1.0, 2.0])), jnp.array([-0.5, -1.0]))
    assert field(0.0, 2.0) == -1.0


def test_integrate_exponential_decay() -> None:
    ts = jnp.linspace(0, 10, 5)
    soln = gf.integrate_field(
        gf.CustomField(decay), jnp.array([1.0]), ts, solver=SOLVER
    )
    assert jnp.allclose(soln.ys[:, 0], jnp.exp(-K * ts), rtol=1e-6)


def test_integrate_backwards() -> None:
    ts = jnp.linspace(10, 0, 5)
    y0 = jnp.array([jnp.exp(-3.0)])
    soln = gf.integrate_field(gf.CustomField(decay), y0, ts, solver=SOLVER)
    assert jnp.allclose(soln.ys[:, 0], jnp.exp(-K * ts), rtol=1e-6)


def test_pytree_state_harmonic_oscillator() -> None:
    field = gf.CustomField(lambda t, y, args: (y[1], -y[0]))
    y0 = (jnp.array(1.0), jnp.array(0.0))
    ts = jnp.linspace(0, 2 * jnp.pi, 9)
    soln = gf.integrate_field(field, y0, ts, solver=SOLVER)
    assert jnp.allclose(soln.ys[0], jnp.cos(ts), atol=1e-6)
    assert jnp.allclose(soln.ys[1], -jnp.sin(ts), atol=1e-6)


def test_args_are_passed() -> None:
    field = gf.CustomField(lambda t, y, args: -args["k"] * y)
    ts = jnp.linspace(0, 10, 5)
    soln = gf.integrate_field(field, jnp.array([1.0]), ts, args={"k": K}, solver=SOLVER)
    assert jnp.allclose(soln.ys[:, 0], jnp.exp(-K * ts), rtol=1e-6)


def test_quantity_leaves_use_their_own_dimension() -> None:
    field = gf.CustomField(
        lambda t, y, args: (jnp.zeros_like(y[0]), jnp.zeros_like(y[1])),
        units="galactic",
    )
    y0 = (u.Q([1.0], "kpc"), u.Q([977.79222168], "km/s"))
    ts = u.Q(jnp.linspace(0, 1, 3), "Myr")
    soln = gf.integrate_field(field, y0, ts, solver=SOLVER)
    assert jnp.allclose(soln.ys[0], 1.0, rtol=1e-6)  # kpc
    assert jnp.allclose(soln.ys[1], 1.0, rtol=1e-6)  # kpc/Myr


def test_diffeqsolver_accepts_custom_field() -> None:
    field = gf.CustomField(decay)
    solver = dfxtra.DiffEqSolver(
        dfx.Dopri8(), stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-8)
    )
    soln = solver(field, t0=0.0, t1=1.0, dt0=None, y0=jnp.array([1.0]))
    assert isinstance(soln, dfx.Solution)
    assert jnp.allclose(soln.ys[-1, 0], jnp.exp(-K), rtol=1e-6)


def test_diffeqsolver_dt0_quantity_or_number() -> None:
    """A Quantity ``dt0`` is converted; a number is in the field's time unit."""
    solver = dfxtra.DiffEqSolver(
        dfx.Euler(), stepsize_controller=dfx.ConstantStepSize()
    )
    y0 = jnp.array([1.0])

    field = gf.CustomField(decay)  # galactic: Myr
    want = solver(field, t0=0.0, t1=1.0, dt0=0.1, y0=y0).ys
    got = solver(field, t0=0.0, t1=1.0, dt0=u.Q(100, "kyr"), y0=y0).ys
    assert jnp.array_equal(got, want)

    # A field without a time unit still takes a plain-number dt0.
    field = gf.CustomField(decay, units="dimensionless")
    got = solver(field, t0=0.0, t1=1.0, dt0=0.1, y0=y0).ys
    assert jnp.array_equal(got, want)

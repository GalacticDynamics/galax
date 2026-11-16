"""Tests for `galax.dynamics.cluster.MassSolver`."""

import diffrax as dfx
import jax.numpy as jnp
import pytest

import unxt as u

import galax.dynamics.cluster as gdc


@pytest.mark.parametrize("dt0", [u.Q(0.1, "Myr"), 1e5], ids=["quantity", "float"])
def test_solve_reads_inputs_in_field_units(dt0):
    """A field in other units than the solver's gives the same mass history.

    The solver is in galactic units (Myr) and the field in solarsystem units
    (yr). A float ``dt0`` is in the field's time unit, so ``1e5`` is 0.1 Myr.
    """
    solver = gdc.MassSolver(
        dfx.Euler(), stepsize_controller=dfx.ConstantStepSize(), event=None
    )
    t0, t1, M0 = u.Q(0.0, "Myr"), u.Q(1.0, "Myr"), u.Q(1e4, "Msun")

    # The same rate, 0.1 / Myr, written in each unit system.
    field = gdc.CustomMassRateField(lambda _t, M, _args: -1e-7 * M, units="solarsystem")
    ref_field = gdc.CustomMassRateField(lambda _t, M, _args: -0.1 * M, units="galactic")

    soln = solver.solve(field, M0, t0, t1, dt0=dt0)
    ref = solver.solve(ref_field, M0, t0, t1, dt0=u.Q(0.1, "Myr"))

    assert int(soln.stats["num_steps"]) == 10
    assert jnp.allclose(soln.ys, ref.ys)
    assert jnp.allclose(ref.ys, 1e4 * 0.99**10)


def test_init_run_read_inputs_in_field_units():
    """`init` and `run` also read times and mass in the field's units."""
    solver = gdc.MassSolver(
        dfx.Euler(), stepsize_controller=dfx.ConstantStepSize(), event=None
    )
    field = gdc.CustomMassRateField(lambda _t, M, _args: -1e-7 * M, units="solarsystem")

    state = solver.init(field, u.Q(1e4, "Msun"), u.Q(0.0, "Myr"), {})
    state = solver.run(field, state, u.Q(1.0, "Myr"), {}, dt0=u.Q(0.1, "Myr"))

    assert jnp.allclose(state.t, 1e6)  # yr
    assert jnp.allclose(state.y, 1e4 * 0.99**10)


def test_step_reads_t1_in_field_units():
    """`step` reads ``t1`` in the field's units."""
    solver = gdc.MassSolver(
        dfx.Euler(), stepsize_controller=dfx.ConstantStepSize(), event=None
    )
    field = gdc.CustomMassRateField(lambda _t, M, _args: -1e-7 * M, units="solarsystem")

    state = solver.init(field, u.Q(1e4, "Msun"), u.Q(0.0, "Myr"), {})
    state = solver.step(field, state, u.Q(0.1, "Myr"), {})

    assert jnp.allclose(state.t, 1e5)  # yr
    assert jnp.allclose(state.y, 1e4 * 0.99)

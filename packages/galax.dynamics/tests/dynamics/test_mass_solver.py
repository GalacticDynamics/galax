"""Tests for `galax.dynamics.cluster.MassSolver`."""

import diffrax as dfx
import jax.numpy as jnp
import pytest

import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.dynamics.cluster as gdc
import galax.potential as gp


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


def test_run_supplies_units_to_the_default_event():
    """`run` gives the default `MassBelowThreshold` event the field's units."""
    solver = gdc.MassSolver(dfx.Euler(), stepsize_controller=dfx.ConstantStepSize())
    field = gdc.CustomMassRateField(lambda _t, M, _args: -1e-7 * M, units="solarsystem")

    state = solver.init(field, u.Q(1e4, "Msun"), u.Q(0.0, "Myr"), {})
    state = solver.run(field, state, u.Q(1.0, "Myr"), {}, dt0=u.Q(0.1, "Myr"))

    assert jnp.allclose(state.y, 1e4 * 0.99**10)


def test_solve_event_reads_mass_in_field_units():
    """The event compares the mass in the field's units, whatever ``args`` says.

    The threshold is 9500 Msun. With ``args["units"]`` in kg the event would read
    the mass (in Msun) as kg and stop at once.
    """
    event = dfx.Event(gdc.MassBelowThreshold(u.Q(9500.0, "Msun")))
    solver = gdc.MassSolver(
        dfx.Euler(), stepsize_controller=dfx.ConstantStepSize(), event=event
    )
    field = gdc.CustomMassRateField(lambda _t, M, _args: -0.1 * M, units="galactic")
    args = {"units": u.unitsystems.si}

    soln = solver.solve(
        field,
        u.Q(1e4, "Msun"),
        u.Q(0.0, "Myr"),
        u.Q(1.0, "Myr"),
        dt0=u.Q(0.1, "Myr"),
        args=args,
    )

    # Stops on the step the mass drops below 9500 Msun: 1e4 * 0.99**6 < 9500.
    assert jnp.allclose(soln.ts, 0.6)
    assert args == {"units": u.unitsystems.si}  # the caller's dict is untouched


def test_baumgardt_reads_inputs_in_its_own_units():
    """`Baumgardt1998MassLossRate` uses its own units, not the potential's."""
    pot = gp.MilkyWayPotential2022()  # galactic units
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q([8, 0, 0], "kpc"), p=u.Q([0, 180, 0], "km/s"), t=u.Q(0, "Gyr")
    )
    orbit = gd.evaluate_orbit(pot, w0, u.Q([0, 1], "Gyr"), dense=True)
    params = {
        "orbit": orbit,
        "potential": pot,
        "m_avg": u.Q(3, "Msun"),
        "xi0": 0.001,
        "alpha": 14.9,
        "r_hm": u.Q(1, "pc"),
    }

    gal = gdc.Baumgardt1998MassLossRate(units="galactic")  # Msun, Myr
    ss = gdc.Baumgardt1998MassLossRate(units="solarsystem")  # Msun, yr

    # The same instant, 500 Myr, in each field's time unit. The rate is per yr.
    assert jnp.allclose(ss(5e8, 1e4, params), gal(500.0, 1e4, params) / 1e6)

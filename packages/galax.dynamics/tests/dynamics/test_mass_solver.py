"""Tests for `galax.dynamics.cluster.MassSolver`."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
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


def test_run_and_solve_accept_args_none():
    """``args=None`` still works now that the solver adds the units to it."""
    solver = gdc.MassSolver(
        dfx.Euler(), stepsize_controller=dfx.ConstantStepSize(), event=None
    )
    field = gdc.CustomMassRateField(lambda _t, M, _args: -0.1 * M, units="galactic")
    M0, t0, t1 = u.Q(1e4, "Msun"), u.Q(0.0, "Myr"), u.Q(1.0, "Myr")

    state = solver.init(field, M0, t0, None)
    state = solver.run(field, state, t1, None, dt0=u.Q(0.1, "Myr"))
    soln = solver.solve(field, M0, t0, t1, dt0=u.Q(0.1, "Myr"), args=None)

    assert jnp.allclose(state.y, 1e4 * 0.99**10)
    assert jnp.allclose(soln.ys, 1e4 * 0.99**10)


def test_default_event_finds_the_root_for_a_field_in_kg():
    """The default event stops a field in kg where it stops one in Msun.

    The cluster loses 20 Msun / Myr from 1e4 Msun, so it is empty at 500 Myr.
    In kg the mass is ~1e34, which the root finder's absolute tolerance on the
    event value could not reach.
    """
    solver = gdc.MassSolver()
    rate = u.Q(-20.0, "Msun/Myr").ustrip("kg/Myr")
    field = gdc.CustomMassRateField(
        lambda _t, _M, _args: rate, units=u.unitsystem("kpc", "Myr", "kg", "rad")
    )

    soln = solver.solve(field, u.Q(1e4, "Msun"), u.Q(0.0, "Myr"), u.Q(1e3, "Myr"))

    assert jnp.allclose(soln.ts, 500.0)


def test_run_passes_non_dict_args_through():
    """A field (and custom event) may take non-dict ``args``, e.g. a tuple."""
    solver = gdc.MassSolver(
        dfx.Euler(), stepsize_controller=dfx.ConstantStepSize(), event=None
    )
    field = gdc.CustomMassRateField(lambda _t, M, args: -args[0] * M, units="galactic")
    M0, t0, t1 = u.Q(1e4, "Msun"), u.Q(0.0, "Myr"), u.Q(1.0, "Myr")

    state = solver.init(field, M0, t0, (0.1,))
    state = solver.run(field, state, t1, (0.1,), dt0=u.Q(0.1, "Myr"))
    soln = solver.solve(field, M0, t0, t1, dt0=u.Q(0.1, "Myr"), args=(0.1,))

    assert jnp.allclose(state.y, 1e4 * 0.99**10)
    assert jnp.allclose(soln.ys, 1e4 * 0.99**10)


def test_default_root_finder_handles_a_threshold_in_kg():
    """A nonzero threshold in kg stops a field in kg at the right time.

    The cluster loses 20 Msun / Myr from 1e4 Msun, so it reaches 5e3 Msun at
    250 Myr.
    """
    threshold = u.Q(5e3, "Msun").uconvert("kg")
    event = dfx.Event(
        gdc.MassBelowThreshold(threshold),
        root_finder=gdc.MassSolver().event.root_finder,
    )
    solver = gdc.MassSolver(event=event)
    rate = u.Q(-20.0, "Msun/Myr").ustrip("kg/Myr")
    field = gdc.CustomMassRateField(
        lambda _t, _M, _args: rate, units=u.unitsystem("kpc", "Myr", "kg", "rad")
    )

    soln = solver.solve(field, u.Q(1e4, "Msun"), u.Q(0.0, "Myr"), u.Q(1e3, "Myr"))

    assert jnp.allclose(soln.ts, 250.0)


@pytest.mark.parametrize("args", [None, {}, (), []], ids=repr)
def test_default_event_accepts_empty_args(args):
    """No args, in any of these spellings, still get the field's units."""
    field = gdc.CustomMassRateField(lambda _t, _M, _args: -20.0, units="galactic")

    soln = gdc.MassSolver().solve(
        field, u.Q(1e4, "Msun"), u.Q(0.0, "Myr"), u.Q(1e3, "Myr"), args=args
    )

    assert jnp.allclose(soln.ts, 500.0)


def test_run_reports_the_event_time():
    """After the event stops the solve, the state's time is the event time."""
    solver = gdc.MassSolver()
    field = gdc.CustomMassRateField(lambda _t, _M, _args: -20.0, units="galactic")

    state = solver.init(field, u.Q(1e4, "Msun"), u.Q(0.0, "Myr"), None)
    state = solver.run(field, state, u.Q(1e3, "Myr"), None)

    assert jnp.allclose(state.t, 500.0)


@pytest.mark.parametrize("M0", [1e-3, 1e4, 1e12, 1e15], ids=lambda m: f"{m:g}")
def test_default_event_at_any_mass_scale(M0):
    """The default root-find works whatever the cluster's mass."""
    field = gdc.CustomMassRateField(lambda _t, _M, _args: -M0 / 500, units="galactic")

    soln = gdc.MassSolver().solve(
        field, u.Q(M0, "Msun"), u.Q(0.0, "Myr"), u.Q(1e3, "Myr")
    )

    assert jnp.allclose(soln.ts, 500.0)


def test_default_event_in_float32():
    """The default root-find also works with 32-bit floats."""
    with jax.enable_x64(False):  # noqa: FBT003
        field = gdc.CustomMassRateField(lambda _t, _M, _args: -20.0, units="galactic")
        soln = gdc.MassSolver().solve(
            field, u.Q(1e4, "Msun"), u.Q(0.0, "Myr"), u.Q(1e3, "Myr")
        )

    assert soln.ts.dtype == jnp.float32
    assert jnp.allclose(soln.ts, 500.0, rtol=1e-4)


def test_run_can_be_chained():
    """`run` returns a state that can be run again."""
    solver = gdc.MassSolver()
    field = gdc.CustomMassRateField(lambda _t, _M, _args: -20.0, units="galactic")

    state = solver.init(field, u.Q(1e4, "Msun"), u.Q(0.0, "Myr"), None)
    state = solver.run(field, state, u.Q(200.0, "Myr"), None)
    assert state.y.shape == ()
    assert jnp.allclose(state.y, 6e3)

    state = solver.run(field, state, u.Q(1e3, "Myr"), None)
    assert jnp.allclose(state.t, 500.0)


@pytest.mark.parametrize(
    ("x64", "M0", "threshold"),
    [(True, 1e-6, 1e6), (False, 1e-2, 1.0)],
    ids=["float64", "float32"],
)
def test_default_root_finder_with_a_threshold_far_above_the_start(x64, M0, threshold):
    """A growing mass crosses a threshold far above where it started."""
    rate = float(np.log(threshold / M0) / 50)  # crosses at 50 Myr
    with jax.enable_x64(x64):
        event = dfx.Event(
            gdc.MassBelowThreshold(u.Q(threshold, "Msun")),
            root_finder=gdc.MassSolver().event.root_finder,
        )
        field = gdc.CustomMassRateField(lambda _t, M, _args: rate * M, units="galactic")
        soln = gdc.MassSolver(event=event).solve(
            field, u.Q(M0, "Msun"), u.Q(0.0, "Myr"), u.Q(100.0, "Myr")
        )

    assert jnp.allclose(soln.ts, 50.0, rtol=1e-3)


def test_default_root_finder_ignores_a_threshold_that_overflows():
    """A threshold too big for float32 in kg doesn't break the solve."""
    with jax.enable_x64(False):  # noqa: FBT003
        event = dfx.Event(
            gdc.MassBelowThreshold(u.Q(1e10, "Msun")),
            root_finder=gdc.MassSolver().event.root_finder,
        )
        field = gdc.CustomMassRateField(
            lambda _t, M, _args: 0 * M, units=u.unitsystem("kpc", "Myr", "kg", "rad")
        )
        soln = gdc.MassSolver(event=event).solve(
            field, u.Q(1e4, "Msun"), u.Q(0.0, "Myr"), u.Q(100.0, "Myr")
        )

    assert jnp.allclose(soln.ts, 100.0)

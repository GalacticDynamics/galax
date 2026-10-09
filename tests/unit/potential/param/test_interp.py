"""Test :mod:`galax.potential._src.params.interp`."""

import equinox as eqx
import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

from galax.potential.params import time_interpolated_parameter

TS = u.Q(jnp.asarray([0.0, 1.0, 2.0, 3.0]), "Gyr")


def _linear(slope: float = 2.0):
    """Build a parameter exactly linear in ``t``, so interpolation is exact."""
    values = u.Q(1.0 + slope * u.ustrip(u.unit("Gyr"), TS), "Msun")
    return time_interpolated_parameter(TS, values), slope


def test_returns_the_tabulated_value_at_a_knot() -> None:
    """On a knot the interpolant must return that knot's value exactly."""
    p, slope = _linear()
    for i, tv in enumerate((0.0, 1.0, 2.0, 3.0)):
        got = u.ustrip(u.unit("Msun"), p(u.Q(tv, "Gyr")))
        assert float(got) == pytest.approx(1.0 + slope * tv, rel=1e-12), i


def test_interpolates_between_knots() -> None:
    """A cubic through linear data must reproduce the line, not the nearest knot.

    Chosen so that "return the nearest knot" and "return the mean of the
    bracketing knots" both give visibly wrong answers.
    """
    p, slope = _linear()
    for tv in (0.25, 1.4, 2.9):
        got = float(u.ustrip(u.unit("Msun"), p(u.Q(tv, "Gyr"))))
        assert got == pytest.approx(1.0 + slope * tv, rel=1e-10), tv


def test_clamps_outside_the_grid() -> None:
    """Outside the tabulated range the value saturates at the boundary.

    Not extrapolated: continuing the edge cubic past the grid invents
    structure the tabulation knows nothing about. A linear ramp makes the
    distinction unmistakable -- extrapolation would keep climbing.
    """
    p, slope = _linear()
    lo = float(u.ustrip(u.unit("Msun"), p(u.Q(-50.0, "Gyr"))))
    hi = float(u.ustrip(u.unit("Msun"), p(u.Q(+50.0, "Gyr"))))
    assert lo == pytest.approx(1.0, rel=1e-12)
    assert hi == pytest.approx(1.0 + slope * 3.0, rel=1e-12)


def test_interpolates_array_valued_parameters() -> None:
    """Time is the leading axis; every trailing axis is carried along."""
    values = u.Q(jnp.stack([jnp.full((2, 3), float(i)) for i in range(4)]), "Msun")
    p = time_interpolated_parameter(TS, values)
    got = u.ustrip(u.unit("Msun"), p(u.Q(1.5, "Gyr")))
    assert got.shape == (2, 3)
    assert jnp.allclose(got, 1.5)


def test_accepts_a_time_in_other_units() -> None:
    """``t`` is converted to the grid's unit, not assumed to share it."""
    p, slope = _linear()
    got = float(u.ustrip(u.unit("Msun"), p(u.Q(1500.0, "Myr"))))
    assert got == pytest.approx(1.0 + slope * 1.5, rel=1e-10)


def test_is_jittable_and_differentiable() -> None:
    """It is evaluated inside an integrator's scan body and differentiated."""
    p, slope = _linear()

    def f(tv):
        return u.ustrip(u.unit("Msun"), p(u.Q(tv, "Gyr")))

    assert float(jax.jit(f)(jnp.asarray(1.25))) == pytest.approx(
        1.0 + slope * 1.25, rel=1e-10
    )
    # d/dt of a line is its slope, and the clamp makes it zero outside.
    assert float(jax.grad(f)(jnp.asarray(1.25))) == pytest.approx(slope, rel=1e-8)
    assert float(jax.grad(f)(jnp.asarray(99.0))) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("ts", "values", "match"),
    [
        (
            u.Q(jnp.zeros((2, 2)), "Gyr"),
            u.Q(jnp.zeros((2, 2)), "Msun"),
            "must be 1-D",
        ),
        (
            u.Q(jnp.asarray([0.0]), "Gyr"),
            u.Q(jnp.asarray([1.0]), "Msun"),
            "at least 2 entries",
        ),
        (
            u.Q(jnp.asarray([0.0, 1.0]), "Gyr"),
            u.Q(jnp.asarray([1.0, 2.0, 3.0]), "Msun"),
            "time on the leading axis",
        ),
    ],
)
def test_the_factory_rejects_mismatched_inputs(ts, values, match: str) -> None:
    """The grid and the table must agree, and there must be something to interpolate."""
    with pytest.raises(ValueError, match=match):
        time_interpolated_parameter(ts, values)


@pytest.mark.parametrize("time_unit", ["Gyr", "Myr"])
def test_the_knot_derivatives_carry_the_right_dimension(time_unit: str) -> None:
    """``derivs`` is d(values)/d(ts), so it is ``values.unit / ts.unit``.

    REGRESSION: it was labelled ``values.unit``. The array was right and the
    interpolated answer was right -- `_interpolate` reads raw ``.value`` for
    the grid, the values and the derivatives alike, so the label never
    reached it -- but the stored quantity was dimensionally a lie, and
    exactly backwards about it: converting to the correct ``Msun/Gyr``
    raised, while converting to the wrong ``Msun`` silently succeeded.

    Parametrized over the time unit because the bug is invisible unless the
    label is compared against ``ts``: with a single unit in play, any wrong
    label is a fixed wrong label.
    """
    slope = 3.0  # Msun per `time_unit`
    ts = u.Q(jnp.asarray([0.0, 1.0, 2.0, 3.0]), time_unit)
    values = u.Q(1.0 + slope * u.ustrip(u.unit(time_unit), ts), "Msun")

    p = time_interpolated_parameter(ts, values)
    derivs = p.args[2]

    assert derivs.unit == values.unit / ts.unit
    assert jnp.allclose(u.ustrip(u.unit(f"Msun/{time_unit}"), derivs), slope)
    # The answer is unchanged by the relabelling.
    assert jnp.allclose(
        u.ustrip(u.unit("Msun"), p(u.Q(1.5, time_unit))), 1.0 + slope * 1.5
    )


@pytest.mark.parametrize(
    "ts",
    [
        u.Q(jnp.asarray([0.0, 2.0, 1.0]), "Gyr"),
        u.Q(jnp.asarray([0.0, 1.0, 1.0]), "Gyr"),
        u.Q(jnp.asarray([0.0, 1.0, jnp.inf]), "Gyr"),
        u.Q(jnp.asarray([0.0, 1.0, jnp.nan]), "Gyr"),
    ],
    ids=["unsorted", "duplicate", "inf", "nan"],
)
def test_the_factory_rejects_a_grid_it_cannot_search(ts) -> None:
    """An unsearchable grid must fail, not return a plausible wrong number.

    REGRESSION: only `MultipoleProfilePotential` checked this, so the public
    factory would accept ``ts = [0, 2, 1]``, build, and answer ``2.0`` at
    ``t = 1.5`` where the table says ``2.5`` -- silently, because
    `eval_log_spline` brackets with a sorted-grid search and an out-of-order
    grid simply selects the wrong interval.

    Finiteness is a separate condition from monotonicity, not a special case
    of it: ``inf`` *passes* ``diff > 0``, since ``inf - 1`` is ``inf``, and
    then every interpolated value is ``nan``. ``nan`` is caught by the
    monotonicity test already, because no comparison involving it is true.

    `eqx.error_if` rather than a Python `if`, since unlike the shape checks
    this compares values, which cannot be branched on under trace.
    """
    values = u.Q(jnp.asarray([1.0, 2.0, 3.0]), "Msun")
    # `eqx.error_if` fires here, at construction: the factory is not jitted,
    # so the condition is concrete and raises before anything is returned.
    with pytest.raises(eqx.EquinoxRuntimeError, match="strictly increasing"):
        time_interpolated_parameter(ts, values)


def test_the_factory_still_accepts_a_sorted_grid() -> None:
    """The guard above must not reject what it is meant to allow."""
    p = time_interpolated_parameter(
        u.Q(jnp.asarray([0.0, 1.0, 2.0]), "Gyr"),
        u.Q(jnp.asarray([1.0, 2.0, 3.0]), "Msun"),
    )
    assert jnp.allclose(u.ustrip(u.unit("Msun"), p(u.Q(0.5, "Gyr"))), 1.5)


def test_the_gradient_at_a_boundary_knot_is_the_interior_slope() -> None:
    """``jax.grad`` at ``ts[0]`` / ``ts[-1]`` must not be half the slope.

    REGRESSION: the clamp was `jnp.clip`, and JAX gives `clip` the
    subgradient 0.5 at a tie, so differentiating at either end knot returned
    exactly *half* the interior slope. Values were never affected -- this is
    gradients with respect to time only.

    It is not an exotic query: ``t = 0`` is galax's default time and grids
    routinely start there, so the first knot is a boundary a caller lands on
    by default rather than by accident.

    Written with `jnp.where`, the endpoints take the interior branch and get
    the interior derivative. That is the useful convention: the clamp exists
    to stop extrapolation, not to claim the parameter goes flat at its last
    knot. Outside the range the gradient is 0, which *is* that claim and is
    correct there.
    """
    ts = u.Q(jnp.asarray([0.0, 1.0, 2.0]), "Gyr")
    slope = 1e12  # Msun per Gyr, exactly linear so the slope is unambiguous
    values = u.Q(1e12 + slope * u.ustrip(u.unit("Gyr"), ts), "Msun")
    p = time_interpolated_parameter(ts, values)

    grad = jax.grad(lambda x: u.ustrip(u.unit("Msun"), p(u.Q(x, "Gyr"))))

    for t_edge in (0.0, 2.0):
        assert float(grad(t_edge)) == pytest.approx(slope, rel=1e-10), t_edge
    assert float(grad(0.5)) == pytest.approx(slope, rel=1e-10)
    # Outside, the value is clamped and the gradient is genuinely zero.
    assert float(grad(-1.0)) == 0.0
    assert float(grad(3.0)) == 0.0


@pytest.mark.parametrize(
    ("ts", "values", "which"),
    [
        (jnp.asarray([0.0, 1.0, 2.0]), jnp.asarray([1.0, 2.0, 3.0]), "ts"),
        (u.Q(jnp.asarray([0.0, 1.0, 2.0]), "Gyr"), [1.0, 2.0, 3.0], "values"),
    ],
    ids=["ts-bare", "values-bare"],
)
def test_a_unitless_argument_says_which_one(ts, values, which: str) -> None:
    """The commonest wrong input must name itself.

    The converter's own failure is ``TypeError: from_() missing 1 required
    keyword-only argument: 'unit'``, which names neither the argument nor
    what it wanted, and is raised identically whichever of the two is at
    fault.
    """
    with pytest.raises(TypeError, match=f"{which} must carry units"):
        time_interpolated_parameter(ts, values)

"""Tests for the radial spline representation."""

from jaxtyping import Array, Float

import interpax
import pytest

import quaxed.numpy as jnp

from galax.potential._src.harmonic.spline import (
    eval_log_spline,
    fit_log_spline,
    radial_grid,
)


def test_radial_grid_endpoints_and_log_spacing() -> None:
    r = radial_grid(65, jnp.asarray(1e-2), jnp.asarray(1e2))
    assert r.shape == (65,)
    assert jnp.isclose(r[0], 1e-2, rtol=1e-14)
    assert jnp.isclose(r[-1], 1e2, rtol=1e-14)
    dlog = jnp.diff(jnp.log(r))
    assert jnp.allclose(dlog, dlog[0], rtol=1e-12)


@pytest.mark.parametrize(
    "x",
    [jnp.linspace(0.0, 1.0, 17), jnp.geomspace(0.05, 1.0, 17)],
    ids=["uniform", "log-spaced"],
)
def test_spline_helpers_reproduce_the_cubic_spline(x: Float[Array, "n_r"]) -> None:
    """`approx_df` + `eval_log_spline` == `CubicSpline(bc_type="not-a-knot")`.

    Pins the equality against `interpax` changes, since a refactor is
    expected there.

    Both spacings are checked: `eval_log_spline` locates its interval with
    `searchsorted` rather than by division, so nothing assumes uniform knots,
    and log-spaced is the production case.
    """
    y = jnp.stack([jnp.sin(3.0 * x), jnp.cos(2.0 * x)], axis=-1)
    xq = jnp.linspace(x[0], x[-1], 51)

    got = eval_log_spline(x, y, fit_log_spline(x, y), xq)
    expect = interpax.CubicSpline(x, y, axis=0, bc_type="not-a-knot", check=False)(xq)

    assert got.shape == (51, 2)
    assert jnp.allclose(got, expect, rtol=0.0, atol=1e-14)


def test_spline_interpolates_its_knots_exactly() -> None:
    x = jnp.linspace(-1.0, 2.0, 21)
    y = (x**3 - x)[:, None]
    got = eval_log_spline(x, y, fit_log_spline(x, y), x)
    assert jnp.allclose(got, y, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("rest", [(), (2,), (2, 3)])
def test_scalar_query_matches_the_batched_path(rest: tuple[int, ...]) -> None:
    """A scalar ``log_rq`` must agree with the same query batched.

    This is the hot path -- `eval_log_spline` runs inside a `diffrax` scan
    body, one query at a time -- and it is the only path where the basis
    factors are rank-0, so the trailing-axis reshape is what differs.
    """
    x = jnp.linspace(0.0, 1.0, 17)
    y = jnp.sin(3.0 * x).reshape((17, *(1,) * len(rest))) * jnp.ones(rest)
    derivs = fit_log_spline(x, y)

    got = eval_log_spline(x, y, derivs, jnp.asarray(0.371))
    expect = eval_log_spline(x, y, derivs, jnp.asarray([0.371]))[0]

    assert got.shape == rest
    assert jnp.array_equal(got, expect)

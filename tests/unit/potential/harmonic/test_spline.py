"""Tests for the radial spline representation."""

import interpax

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


def test_spline_helpers_reproduce_natural_cubic_spline() -> None:
    """`approx_df` + `CubicHermiteSpline` == `CubicSpline(bc_type="natural")`.

    Verified exact (0.0) during design; this pins it against `interpax`
    changes, since a refactor there is expected.
    """
    x = jnp.linspace(0.0, 1.0, 17)
    y = jnp.stack([jnp.sin(3.0 * x), jnp.cos(2.0 * x)], axis=-1)
    xq = jnp.linspace(0.0, 1.0, 51)

    got = eval_log_spline(x, y, fit_log_spline(x, y), xq)
    expect = interpax.CubicSpline(x, y, axis=0, bc_type="natural", check=False)(xq)

    assert got.shape == (51, 2)
    assert jnp.allclose(got, expect, atol=1e-14)


def test_spline_interpolates_its_knots_exactly() -> None:
    x = jnp.linspace(-1.0, 2.0, 21)
    y = (x**3 - x)[:, None]
    got = eval_log_spline(x, y, fit_log_spline(x, y), x)
    assert jnp.allclose(got, y, atol=1e-12)

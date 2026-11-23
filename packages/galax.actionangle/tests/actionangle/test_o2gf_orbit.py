"""Tests for O2GF orbit circulating_axes and alignment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from galax.actionangle._src._o2gf_orbit import (
    align_circulation_with_z,
    circulating_axes,
    is_loop,
)

T = jnp.linspace(0.0, 40.0, 400)
C, S, Z = jnp.cos(T), jnp.sin(T), jnp.zeros_like(T)


def _orbit(x, y, z):
    q = jnp.stack([x, y, z], axis=1)
    # velocities are the analytic time derivatives for the planar circle and box
    return q, jnp.gradient(q, T, axis=0)


ORBITS = {
    "xy": (_orbit(C, S, Z), (False, False, True), 2),
    "yz": (_orbit(Z, C, S), (True, False, False), 0),
    "xz": (_orbit(C, Z, S), (False, True, False), 1),
    "box": (
        _orbit(C, 0.7 * jnp.cos(1.3 * T), 0.5 * jnp.cos(1.7 * T)),
        (False, False, False),
        2,
    ),
}


def _check(q, p, circ, expected, axis_exp, out):
    q2, p2, axis = out
    assert tuple(np.asarray(circ)) == expected
    assert bool(is_loop(circ)) == any(expected)
    assert int(axis) == axis_exp
    if any(expected):
        L = jnp.cross(q2, p2)
        assert jnp.all(L[:, 2] * L[0, 2] > 0)
        assert jnp.allclose(L[:, :2], 0, atol=1e-6)


@pytest.mark.parametrize("name", list(ORBITS))
@pytest.mark.parametrize("jit", [False, True])
def test_single_orbit(name, jit):
    (q, p), expected, axis_exp = ORBITS[name]
    circ_f, align_f = (
        (jax.jit(circulating_axes), jax.jit(align_circulation_with_z))
        if jit
        else (circulating_axes, align_circulation_with_z)
    )
    circ = circ_f(q, p)
    _check(q, p, circ, expected, axis_exp, align_f(q, p, circ))


def test_vmap():
    qs = jnp.stack([v[0][0] for v in ORBITS.values()])
    ps = jnp.stack([v[0][1] for v in ORBITS.values()])
    circ = jax.jit(jax.vmap(circulating_axes))(qs, ps)
    out = jax.jit(jax.vmap(align_circulation_with_z))(qs, ps, circ)
    for i, (_, expected, axis_exp) in enumerate(ORBITS.values()):
        _check(qs[i], ps[i], circ[i], expected, axis_exp, tuple(o[i] for o in out))


def test_circulation_skips_nonfinite_samples():
    (q, p), expected, _ = ORBITS["xy"]
    q = q.at[0].set(jnp.nan).at[200].set(jnp.nan)
    assert tuple(np.asarray(circulating_axes(q, p))) == expected

"""O2GF's private circulation check agrees with `galax.orbits.circulation`.

`galax.actionangle` does not depend on `galax.orbits`, so O2GF keeps its own
copy of the check. This test keeps the two from drifting apart.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import galax.orbits as go
from galax.actionangle._src._o2gf_orbit import circulating_axes

T = jnp.linspace(0.0, 40.0, 400)
C, S, Z = jnp.cos(T), jnp.sin(T), jnp.zeros_like(T)


def _orbit(x, y, z):
    q = jnp.stack([x, y, z], axis=1)
    return q, jnp.gradient(q, T, axis=0)


ORBITS = {
    "xy": _orbit(C, S, Z),
    "yz": _orbit(Z, C, S),
    "xz": _orbit(C, Z, S),
    "tilted": _orbit(C, S, 0.3 * S),
    "box": _orbit(C, 0.7 * jnp.cos(1.3 * T), 0.5 * jnp.cos(1.7 * T)),
    "radial": (
        jnp.outer(1 + T, jnp.array([1.1, 2.2, 3.3])),
        jnp.tile(jnp.array([0.7, 1.4, 2.1]), (400, 1)),
    ),
    "nan": (_orbit(C, S, Z)[0].at[5].set(jnp.nan), _orbit(C, S, Z)[1]),
    "all_nan": (jnp.full((400, 3), jnp.nan), _orbit(C, S, Z)[1]),
}


@pytest.mark.parametrize("name", list(ORBITS))
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("scale", [1e-20, 1.0, 3e20])
def test_matches_galax_orbits(name, dtype, scale):
    q, p = ORBITS[name]
    q, p = (scale * q).astype(dtype), p.astype(dtype)
    circ = circulating_axes(q, p, jnp.ones(len(q), bool))
    np.testing.assert_array_equal(circ, go.circulation(q, p))

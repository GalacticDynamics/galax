"""`galax.orbits.specific_angular_momentum` and `galax.orbits.omega`.

`specific_angular_momentum` takes any
`galax.coordinates.AbstractPhaseSpaceObject` and lives in `galax.orbits`, so it
is tested here rather than in the coordinates contract. Testing it there made
the lower portion's tests import the higher one -- the same inversion the
energy functions had before they moved.
"""

import pytest

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.orbits as go


@pytest.fixture
def q() -> u.Quantity:
    """Two positions on the x-axis."""
    return u.Q([[8.0, 0.0, 0.0], [9.0, 0.0, 0.0]], "kpc")


@pytest.fixture
def p() -> u.Quantity:
    """Two velocities along +y, so the motion is planar in x-y."""
    return u.Q([[0.0, 220.0, 0.0], [0.0, 200.0, 0.0]], "km/s")


@pytest.fixture(params=["coordinate", "orbit"])
def w(request, q, p) -> gc.AbstractPhaseSpaceObject:
    """Return the same state as a `PhaseSpaceCoordinate` and as an `Orbit`.

    `Orbit` implements the coordinates contract, so the function must accept
    both.
    """
    t = u.Q([0.0, 0.0], "Myr")
    if request.param == "coordinate":
        return gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
    return go.Orbit(q=q, p=p, t=t, interpolant=None, frame=gc.frames.simulation_frame)


def test_shape_and_type(w) -> None:
    """The result is a Cartesian 3-vector shaped like the positions."""
    h = go.specific_angular_momentum(w)
    assert isinstance(h, cx.vecs.Cartesian3D)
    assert h.shape == w.q.shape


def test_planar_motion_gives_purely_z(w) -> None:
    """For motion in the x-y plane, `q x p` is along +z and nothing else.

    Shape and type alone would pass for any vector-valued function; this pins
    the value, which is what makes the test about angular momentum.
    """
    h = go.specific_angular_momentum(w)
    zero = u.Q(0.0, h.x.unit)
    assert jnp.allclose(h.x, zero, atol=u.Q(1e-8, h.x.unit))
    assert jnp.allclose(h.y, zero, atol=u.Q(1e-8, h.y.unit))
    assert jnp.all(h.z > zero)


def test_omega_value() -> None:
    """`omega` resolves with `galax.orbits` alone and pins the value."""
    x = jnp.array([8.0, 0, 0])
    v = jnp.array([0.0, 8, 0])
    assert float(go.omega(x, v)) == 1.0

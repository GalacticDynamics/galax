"""`galax.dynamics.evaluate_orbit` over a range of potentials.

These assertions used to live on `AbstractPotential_Test`, the contract
inherited by 43 test classes across 42 files in `galax.potential`'s test tree,
where they ran 86 times -- twice for each. That made the *potential* portion's
tests require `galax.dynamics`, so the tree could not collect without the
higher portion installed. It is the inversion #937 removed from the
*coordinates* contract, which carried a `test_angular_momentum` calling
`gd.specific_angular_momentum`; that one was rewritten into this tree rather
than relocated, for the same reason.

They assert properties of the integration -- the returned type, its shape, and
that the requested times come back -- not of any particular force law, so
breadth over all 43 bought little. What it did cover is the kinds of
potential that reach the solver differently, which is what the parametrization
below keeps: a simple analytic profile, a cuspy halo, a composite of several
components, and the degenerate zero-force case.
"""

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.dynamics as gd
import galax.dynamics.custom_types as gt
import galax.potential as gp

# One per kind that reaches the solver differently, not one per potential.
POTENTIALS = {
    "kepler": gp.KeplerPotential(m_tot=u.Q(1e12, "Msun"), units="galactic"),
    "hernquist": gp.HernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(5.0, "kpc"), units="galactic"
    ),
    "nfw": gp.NFWPotential(m=u.Q(1e12, "Msun"), r_s=u.Q(20.0, "kpc"), units="galactic"),
    "composite": gp.MilkyWayPotential(),
    "null": gp.NullPotential(units="galactic"),
}


@pytest.fixture(params=list(POTENTIALS), ids=list(POTENTIALS))
def pot(request: pytest.FixtureRequest) -> gp.AbstractPotential:
    """Return one potential of each kind that reaches the solver differently."""
    return POTENTIALS[request.param]


@pytest.fixture
def xv(pot: gp.AbstractPotential) -> gt.Sz6:
    """Return a bare 6-vector in the potential's own unit system.

    Bare rather than a `Quantity`, which is what `evaluate_orbit` accepts here
    and what the original fixtures built with `jnp.concat([x.value, v.value])`.
    """
    x = u.Q(jnp.asarray([1.0, 2.0, 3.0]), pot.units["length"])
    v = u.Q(jnp.asarray([4.0, 5.0, 6.0]), pot.units["speed"])
    return jnp.concat([x.value, v.value])


@pytest.fixture
def ts() -> gt.QuSzTime:
    """Return the save times."""
    return u.Q(jnp.linspace(0.0, 1.0, 100), "Myr")


def test_evaluate_orbit(pot: gp.AbstractPotential, xv: gt.Sz6, ts: gt.QuSzTime) -> None:
    """A single initial condition gives one orbit over the requested times.

    `t` is the save-times array handed back, not a recomputed one, so it
    compares exactly -- for a batch too, where it stays shared and 1-D.
    """
    orbit = gd.evaluate_orbit(pot, xv, ts)
    assert isinstance(orbit, gd.Orbit)
    assert orbit.shape == (len(ts),)
    assert jnp.array_equal(orbit.t, ts)


def test_evaluate_orbit_batch(
    pot: gp.AbstractPotential, xv: gt.Sz6, ts: gt.QuSzTime
) -> None:
    """A batch of initial conditions keeps its leading shape."""
    orbits = gd.evaluate_orbit(pot, xv[None, :], ts)
    assert isinstance(orbits, gd.Orbit)
    assert orbits.shape == (1, len(ts))
    assert jnp.array_equal(orbits.t, ts)

    orbits = gd.evaluate_orbit(pot, jnp.stack([xv, xv], axis=0), ts)
    assert isinstance(orbits, gd.Orbit)
    assert orbits.shape == (2, len(ts))
    assert jnp.array_equal(orbits.t, ts)

"""Benchmarks for the SCF potential.

NOTE: pytest-codspeed 4.2.0's walltime instrument reports absolute numbers
that are too small by roughly the `iter_per_round` factor -- it prints
`stats.min_ns / iter_per_round` when `BenchmarkStats.from_list` has already
divided by it, so a provably 5.00 ms busy-loop is reported as 0.557 ms. The
dashboard's *relative* comparisons between runs (e.g. this commit vs. main)
remain valid, since the same bias applies to both sides; only the absolute
figures are optimistic.
"""

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp


def _pot(nmax: int, lmax: int) -> gp.SCFPotential:
    Snlm = jnp.zeros((nmax + 1, lmax + 1, lmax + 1)).at[0, 0, 0].set(1.0)
    return gp.SCFPotential(
        m_tot=u.Q(1e12, "Msun"),
        r_s=u.Q(10.0, "kpc"),
        Snlm=Snlm,
        Tnlm=jnp.zeros_like(Snlm),
        units="galactic",
    )


# NOTE: `jax.jit` on a *bound* potential method (e.g. `jax.jit(pot.potential)`)
# hashes `pot` as part of the closure. Equinox's default `Module.__hash__`
# hashes every field, including `constants["G"]`, a JAX-array-valued
# `Quantity` -- which is unhashable. So `pot` must be an explicit (dynamic)
# jit argument, not baked into the jitted closure. This is a pre-existing,
# potential-class-agnostic quirk (reproduces for `HernquistPotential` too),
# not something introduced or fixed by SCF -- worked around here rather than
# touched in production code.
def _potential(pot: gp.SCFPotential, xyz: u.AbstractQuantity, t: u.AbstractQuantity):
    return pot.potential(xyz, t)


def _gradient(pot: gp.SCFPotential, xyz: u.AbstractQuantity, t: u.AbstractQuantity):
    return pot.gradient(xyz, t)


NL = [(2, 2), (6, 4), (12, 6)]
NPOINTS = [1, 1_000, 100_000]


# NOTE: pytest-codspeed's walltime instrument wraps `item.runtest` -- the
# test *call* phase -- and repeats only that. Fixture setup runs once before
# the repeated calls, outside the measured region. So `pot` and `xyz` are
# built here in fixtures, not inside the test bodies: constructing an
# `SCFPotential` costs ~5ms, which used to swamp the ~0.08ms evaluation this
# benchmark is meant to measure.
@pytest.fixture(params=NL, ids=lambda v: f"nl{v}")
def pot(request: pytest.FixtureRequest) -> gp.SCFPotential:
    """SCF potential, built outside the benchmarked region."""
    nmax, lmax = request.param
    return _pot(nmax, lmax)


@pytest.fixture(params=NPOINTS, ids=lambda v: f"n{v}")
def xyz(request: pytest.FixtureRequest) -> u.AbstractQuantity:
    """Random evaluation points, built outside the benchmarked region."""
    key = jax.random.key(0)
    return u.Q(jax.random.normal(key, (request.param, 3)) * 10.0, "kpc")


@pytest.mark.benchmark(group="galax.potential.scf", max_time=1.0)
def test_compile(pot: gp.SCFPotential) -> None:
    """Time to trace and compile the potential."""
    xyz = u.Q(jnp.ones((1_000, 3)), "kpc")
    _ = jax.jit(_potential).lower(pot, xyz, u.Q(0.0, "Gyr")).compile()


@pytest.mark.benchmark(group="galax.potential.scf", max_time=1.0)
def test_potential(pot: gp.SCFPotential, xyz: u.AbstractQuantity) -> None:
    """Evaluate the potential on a batch of positions."""
    t = u.Q(0.0, "Gyr")
    fn = jax.jit(_potential)
    _ = jax.block_until_ready(fn(pot, xyz, t))  # warm up the cache

    _ = jax.block_until_ready(fn(pot, xyz, t))


@pytest.mark.benchmark(group="galax.potential.scf", max_time=1.0)
def test_gradient(pot: gp.SCFPotential, xyz: u.AbstractQuantity) -> None:
    """Evaluate the gradient (autodiff) on a batch of positions."""
    t = u.Q(0.0, "Gyr")
    fn = jax.jit(_gradient)
    _ = jax.block_until_ready(fn(pot, xyz, t))

    _ = jax.block_until_ready(fn(pot, xyz, t))

"""Benchmarks for the SCF potential."""

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


@pytest.mark.parametrize(("nmax", "lmax"), NL, ids=lambda v: f"nl{v}")
@pytest.mark.benchmark(group="galax.potential.scf", max_time=1.0)
def test_compile(nmax, lmax) -> None:
    """Time to trace and compile the potential."""
    pot = _pot(nmax, lmax)
    xyz = u.Q(jnp.ones((1_000, 3)), "kpc")
    _ = jax.jit(_potential).lower(pot, xyz, u.Q(0.0, "Gyr")).compile()


@pytest.mark.parametrize("npoints", NPOINTS, ids=lambda v: f"n{v}")
@pytest.mark.parametrize(("nmax", "lmax"), NL, ids=lambda v: f"nl{v}")
@pytest.mark.benchmark(group="galax.potential.scf", max_time=1.0)
def test_potential(nmax, lmax, npoints) -> None:
    """Evaluate the potential on a batch of positions."""
    pot = _pot(nmax, lmax)
    key = jax.random.key(0)
    xyz = u.Q(jax.random.normal(key, (npoints, 3)) * 10.0, "kpc")
    t = u.Q(0.0, "Gyr")
    fn = jax.jit(_potential)
    _ = jax.block_until_ready(fn(pot, xyz, t))  # warm up the cache

    _ = jax.block_until_ready(fn(pot, xyz, t))


@pytest.mark.parametrize("npoints", NPOINTS, ids=lambda v: f"n{v}")
@pytest.mark.parametrize(("nmax", "lmax"), NL, ids=lambda v: f"nl{v}")
@pytest.mark.benchmark(group="galax.potential.scf", max_time=1.0)
def test_gradient(nmax, lmax, npoints) -> None:
    """Evaluate the gradient (autodiff) on a batch of positions."""
    pot = _pot(nmax, lmax)
    key = jax.random.key(0)
    xyz = u.Q(jax.random.normal(key, (npoints, 3)) * 10.0, "kpc")
    t = u.Q(0.0, "Gyr")
    fn = jax.jit(_gradient)
    _ = jax.block_until_ready(fn(pot, xyz, t))

    _ = jax.block_until_ready(fn(pot, xyz, t))

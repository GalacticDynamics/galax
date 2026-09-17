"""Benchmarks for `MultipoleProfilePotential`.

NOTE: pytest-codspeed 4.2.0's walltime instrument under-reports absolute
numbers by roughly the `iter_per_round` factor; relative comparisons between
runs remain valid. See the note in `test_potential_scf.py`.
"""

from collections.abc import Callable

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp


def _source() -> gp.TriaxialNFWPotential:
    return gp.TriaxialNFWPotential(
        m=u.Q(1e12, "Msun"),
        r_s=u.Q(10.0, "kpc"),
        q1=1.0,
        q2=0.8,
        units="galactic",
    )


def _pot(l_max: int, n_r: int) -> gp.MultipoleProfilePotential:
    return gp.MultipoleProfilePotential.from_potential(
        _source(),
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e4, "kpc"),
        n_r=n_r,
        l_max=l_max,
        symmetry="triaxial",
    )


# NOTE: `pot` must be an explicit jit argument, never baked into the closure --
# equinox's default `Module.__hash__` hashes `constants["G"]`, a JAX-array
# Quantity, which is unhashable. Pre-existing and class-agnostic; see the same
# note in `test_potential_scf.py`.
def _potential(pot, xyz, t):
    return pot.potential(xyz, t)


def _gradient(pot, xyz, t):
    return pot.gradient(xyz, t)


@pytest.mark.parametrize(("l_max", "n_r"), [(0, 128), (4, 128), (8, 128), (8, 512)])
@pytest.mark.benchmark(group="multipole_profile_build")
def test_build(benchmark, l_max: int, n_r: int) -> None:
    """Construction cost: projection + Poisson + spline fitting."""
    benchmark(lambda: _pot(l_max, n_r))


@pytest.mark.parametrize("l_max", [0, 4, 8])
@pytest.mark.parametrize("func", [_potential, _gradient])
@pytest.mark.benchmark(group="multipole_profile_eval")
def test_eval(benchmark, func: Callable, l_max: int) -> None:
    """Per-point evaluation, and the `jax.grad` force cost relative to it."""
    pot = _pot(l_max, 128)
    xyz = u.Q(jnp.asarray([1.0, 2.0, 3.0]), "kpc")
    t = u.Q(0.0, "Myr")
    jitted = jax.jit(func)
    jax.block_until_ready(jitted(pot, xyz, t))
    benchmark(lambda: jax.block_until_ready(jitted(pot, xyz, t)))


@pytest.mark.parametrize("n", [1_000, 100_000])
@pytest.mark.benchmark(group="multipole_profile_batch")
def test_eval_batch(benchmark, n: int) -> None:
    """Batched evaluation, the regime that matters for orbit integration."""
    pot = _pot(8, 128)
    key = jax.random.PRNGKey(0)
    xyz = u.Q(10.0 * jax.random.normal(key, (n, 3)), "kpc")
    t = u.Q(0.0, "Myr")
    jitted = jax.jit(_gradient)
    jax.block_until_ready(jitted(pot, xyz, t))
    benchmark(lambda: jax.block_until_ready(jitted(pot, xyz, t)))

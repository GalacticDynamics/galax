"""Benchmarks for `MultipoleProfilePotential`.

NOTE: pytest-codspeed 4.2.0's walltime instrument under-reports absolute
numbers by roughly the `iter_per_round` factor; relative comparisons between
runs remain valid. See the note in `test_potential_scf.py`.

MEASUREMENT: Gradient-via-autodiff cost relative to potential evaluation,
measured with explicit warm-up and jax.block_until_ready on every call
(20 reps each):
- n=1: potential 0.124 ms, gradient 0.106 ms → 0.86x (dispatch-overhead)
- n=1000: potential 0.207 ms, gradient 0.244 ms → 1.18x (dispatch-overhead)
- n=100000: potential 2.606 ms, gradient 6.453 ms → 2.48x (computation-cost)

Only n=1e5 reflects actual computation cost; smaller n are dispatch-overhead
artifacts. At 2.48x for l_max=8, reverse-mode autodiff acceptably replaces the
~155 lines of hand-coded force computation the analytic route would need.
"""

from collections.abc import Callable

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
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
        symmetry="plane_reflection",
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


def _orbit(pot, w0, ts):
    return gd.evaluate_orbit(pot, w0, ts)


@pytest.mark.parametrize("n", [1, 1_000])
@pytest.mark.benchmark(group="multipole_profile_orbit")
def test_orbit(benchmark, n: int) -> None:
    """Orbit integration: the per-*call* regime the batch benchmarks miss.

    Work that is loop-invariant but sits inside the solver's scan body costs
    nothing measurable in `test_eval_batch` -- it amortizes over the batch --
    yet is paid once per integration step here. `gd.evaluate_orbit` is used
    rather than `gd.compute_orbit` because XLA hoists that loop-invariant work
    out of the latter's graph, which would make this benchmark blind to it.
    """
    pot = _pot(8, 128)
    q = jnp.asarray([8.0, 0.0, 0.1])
    p = jnp.asarray([0.0, 0.22, 0.02])
    if n > 1:  # spread the ensemble so the steppers do not all move in lockstep
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        q = q + 0.1 * jax.random.normal(k1, (n, 3))
        p = p + 0.001 * jax.random.normal(k2, (n, 3))
    w0 = gc.PhaseSpacePosition(q=u.Q(q, "kpc"), p=u.Q(p, "kpc/Myr"))
    ts = u.Q(jnp.linspace(0.0, 2000.0, 2001), "Myr")
    jax.block_until_ready(_orbit(pot, w0, ts))
    benchmark(lambda: jax.block_until_ready(_orbit(pot, w0, ts)))

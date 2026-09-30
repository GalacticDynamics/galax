"""Benchmarks for `MultipoleProfilePotential`.

NOTE: pytest-codspeed 4.2.0's walltime instrument under-reports absolute
numbers by roughly the `iter_per_round` factor; relative comparisons between
runs remain valid. See the note in `test_potential_scf.py`.

MEASUREMENT: gradient cost relative to potential evaluation, l_max=8, float64,
with warm-up and jax.block_until_ready on every call. The two gradient paths
are timed interleaved in one process and reduced with `min`, because this is a
shared machine and a median under contention moved by 50% between runs:

===== ========== ============== ============== =========
n      potential  reverse-mode   `_gradient`    speedup
===== ========== ============== ============== =========
1000   0.17 ms    0.40 ms 2.36x  0.33 ms 1.94x  1.21x
1e5    6.05 ms    14.77 ms 2.44x 11.64 ms 1.93x 1.27x
===== ========== ============== ============== =========

`MultipoleProfileMixin._gradient` replaced `AbstractPotential`'s `jax.grad`
default with the chain rule written out; see `expansion_gradient`. It takes
the ratio from ~2.4x to ~1.9x of a forward evaluation.

Where it shows up is the orbit integration below, which is what the gradient
cost is actually *for* -- n=1000, 2001 steps: 1145.6 ms reverse-mode against
876.1 ms, a 1.31x speedup. n=1 is unchanged at 42.2 vs 42.4 ms, since that
regime is dispatch-bound and not computation-bound.

The saving is structural rather than arithmetic. Reverse mode over the whole
expansion pushes an (n_modes,) cotangent back through the radial spline's
gather, and undoing that gather with a scatter-add is most of the cost: the
radial and angular halves cost only +0.97 ms and +0.79 ms of autodiff overhead
on their own, against +7.16 ms for the two composed. Since log r is one scalar
per position, the radial half is taken forward-mode instead. The angular half
genuinely has n_modes outputs against three inputs and stays in reverse.

Note this is *not* the ~155 lines of hand-coded force computation the analytic
route was assumed to need. The harmonic derivatives are ~10% of the gradient's
cost, so hand-writing them would buy little; both halves here remain `jax`'s
own derivatives of the same code the potential evaluates, with nothing to
drift. `test_gradient_matches_autodiff` pins them together.
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

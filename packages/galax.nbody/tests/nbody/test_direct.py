"""Tests for the direct-sum force backend."""

import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from galax.nbody._src.direct import (
    DirectSum,
    direct_sum_accelerations,
    direct_sum_potential,
)
from galax.nbody._src.kernels import PlummerKernel
from galax.nbody._src.particles import Sources, Targets

G = 1.7


def brute(q_t, q_s, m, h_t, h_s, self_index, G, combine=np.maximum):
    """Return accelerations and potentials from a numpy double loop."""
    acc = np.zeros_like(q_t)
    pot = np.zeros(len(q_t))
    for i in range(len(q_t)):
        for j in range(len(q_s)):
            if j == self_index[i]:
                continue
            d = q_s[j] - q_t[i]
            r2 = d @ d + combine(h_t[i], h_s[j]) ** 2
            acc[i] += G * m[j] * d * r2**-1.5
            pot[i] -= G * m[j] * r2**-0.5
    return acc, pot


def massive(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 3))
    m = rng.uniform(0.5, 2.0, n)
    h = rng.uniform(0.01, 0.1, n)
    s = Sources(q=jnp.asarray(q), m=jnp.asarray(m), h=jnp.asarray(h))
    t = Targets(q=s.q, h=s.h, self_index=jnp.arange(n))
    return s, t


def check_brute(backend, s, t, combine=np.maximum):
    want_a, want_p = brute(
        np.asarray(t.q),
        np.asarray(s.q),
        np.asarray(s.m),
        np.asarray(t.h),
        np.asarray(s.h),
        np.asarray(t.self_index),
        G,
        combine,
    )
    p = backend.prepare(s)
    np.testing.assert_allclose(backend.accelerations(p, t, G=G), want_a, rtol=1e-12)
    np.testing.assert_allclose(backend.potential(p, t, G=G), want_p, rtol=1e-12)


def test_matches_bruteforce_unequal_masses_and_softening() -> None:
    check_brute(DirectSum(chunk_size=3), *massive(7))


def test_tracer_targets_against_sources() -> None:
    s, _ = massive(5)
    rng = np.random.default_rng(1)
    t = Targets(
        q=jnp.asarray(rng.normal(size=(4, 3))),
        h=jnp.asarray(rng.uniform(0.01, 0.1, 4)),
        self_index=jnp.full(4, -1),
    )
    check_brute(DirectSum(chunk_size=2), s, t)


def test_newtons_third_law() -> None:
    s, t = massive(9)
    a = DirectSum(chunk_size=4).accelerations(s, t, G=G)
    total = jnp.sum(s.m[:, None] * a, axis=0)
    scale = jnp.sum(s.m[:, None] * jnp.abs(a))
    np.testing.assert_allclose(total, 0.0, atol=1e-14 * float(scale))


def test_coincident_particles_finite_value_and_grad() -> None:
    m, h, idx = jnp.ones(2), jnp.full(2, 0.05), jnp.arange(2)
    q0 = jnp.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])

    def f(q):
        a = DirectSum().accelerations(Sources(q, m, h), Targets(q, h, idx), G=G)
        return jnp.sum(a**2) + jnp.sum(a)

    assert jnp.all(
        jnp.isfinite(
            DirectSum().accelerations(Sources(q0, m, h), Targets(q0, h, idx), G=G)
        )
    )
    assert jnp.all(jnp.isfinite(jax.grad(f)(q0)))


def test_zero_softening_distinct_points_finite_grad() -> None:
    s, t = massive(4)
    h = jnp.zeros(4)

    def f(q):
        b = DirectSum(chunk_size=3)
        return jnp.sum(
            b.accelerations(Sources(q, s.m, h), Targets(q, h, t.self_index), G=G) ** 2
        )

    assert jnp.all(jnp.isfinite(jax.grad(f)(s.q)))


def test_chunk_not_dividing_n() -> None:
    s, t = massive(7)
    a7 = DirectSum(chunk_size=7).accelerations(s, t, G=G)
    for c in (1, 4):
        np.testing.assert_allclose(
            DirectSum(chunk_size=c).accelerations(s, t, G=G), a7, rtol=1e-13
        )


def test_potential_matches_bruteforce_and_excludes_self() -> None:
    s, t = massive(6)
    check_brute(DirectSum(chunk_size=4), s, t)
    # A lone particle feels no potential from itself.
    one = Sources(q=jnp.zeros((1, 3)), m=jnp.ones(1), h=jnp.zeros(1))
    p = DirectSum().potential(one, Targets(one.q, one.h, jnp.array([0])), G=G)
    assert p[0] == 0.0


def test_combine_override_changes_pair_softening() -> None:
    class MinKernel(PlummerKernel):
        def combine(self, h_i, h_j):
            return jnp.minimum(h_i, h_j)

    check_brute(DirectSum(kernel=MinKernel(), chunk_size=3), *massive(7), np.minimum)


def test_gradient_memory_is_chunk_bounded() -> None:
    n, chunk = 4096, 256
    s, t = massive(n)

    def f(q):
        return jnp.sum(
            direct_sum_accelerations(
                Sources(q, s.m, s.h), t, kernel=PlummerKernel(), G=G, chunk_size=chunk
            )
        )

    mem = jax.jit(jax.grad(f)).lower(s.q).compile().memory_analysis()
    assert mem.temp_size_in_bytes < 16 * chunk * n * 8


def test_check_grads_small() -> None:
    s, t = massive(5)
    b = DirectSum(chunk_size=2)
    check_grads(
        lambda q: b.accelerations(Sources(jnp.asarray(q), s.m, s.h), t, G=G),
        (s.q,),
        order=1,
        modes=("rev",),
    )
    check_grads(
        lambda m: b.accelerations(Sources(s.q, jnp.asarray(m), s.h), t, G=G),
        (s.m,),
        order=1,
        modes=("rev",),
    )


@pytest.mark.slow
def test_runtime_typechecked_grad_policies() -> None:
    """Every gradient policy runs with beartype checking the modules."""
    pytest.importorskip("beartype")
    code = textwrap.dedent("""
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jaxtyping import install_import_hook
        with install_import_hook("galax.nbody._src", "beartype.beartype"):
            from galax.nbody._src.particles import Sources, Targets
            from galax.nbody._src.direct import DirectSum
        s = Sources(q=jnp.eye(3), m=jnp.ones(3), h=jnp.full(3, 0.1))
        t = Targets(q=s.q, h=s.h, self_index=jnp.arange(3))
        for g in ("autodiff", "stop", "none"):
            b = DirectSum(grad=g)
            b.accelerations(s, t, G=1.0)
            b.potential(s, t, G=1.0)
    """)
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr[-3000:]


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_chunk_size_must_be_positive_int(chunk_size) -> None:
    # Under runtime type checking a float fails the annotation first.
    with pytest.raises((ValueError, TypeError), match="chunk_size"):
        DirectSum(chunk_size=chunk_size)


def test_chunk_size_accepts_numpy_ints() -> None:
    assert DirectSum(chunk_size=np.int64(8)).chunk_size == 8
    for bad in (8.0, 0, -1):
        with pytest.raises((ValueError, TypeError), match="chunk_size"):
            DirectSum(chunk_size=bad)


@pytest.mark.parametrize("fn", [direct_sum_accelerations, direct_sum_potential])
def test_non_finite_target_leaves_source_at_origin_finite(fn) -> None:
    """A NaN target's substitute position does not land on an unsoftened source."""
    src_q = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]])

    def loss(src_q, x):
        s = Sources(q=src_q, m=jnp.ones(2), h=jnp.zeros(2))
        t = Targets(q=x, h=jnp.zeros(2), self_index=jnp.asarray([-1, -1]))
        out = fn(s, t, kernel=PlummerKernel(), G=1.0, chunk_size=1)
        return jnp.sum(out[0] ** 2)

    good = jnp.asarray([[1.0, 1.0, 0.0], [3.0, 0.0, 0.0]])
    bad = good.at[1].set(jnp.nan)
    np.testing.assert_allclose(
        jax.grad(loss)(src_q, bad), jax.grad(loss)(src_q, good), rtol=1e-13
    )

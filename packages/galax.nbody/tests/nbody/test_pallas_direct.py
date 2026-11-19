"""Tests for the Pallas direct-sum backend.

Run in interpret mode everywhere, and compiled too when a GPU is present.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import galax.nbody as gn
from .test_direct import G, check_brute, massive
from galax.nbody._src.backend import Sources, Targets
from galax.nbody._src.direct import DirectSum
from galax.nbody._src.kernels import (
    DehnenK1Kernel,
    DehnenK2Kernel,
    NewtonianKernel,
    PlummerKernel,
    SplineKernel,
)
from galax.nbody._src.pallas_direct import PallasDirectSum

MODES = [pytest.param(True, id="interpret")]
if jax.default_backend() == "gpu":
    MODES.append(pytest.param(False, id="compiled"))


@pytest.fixture(params=MODES)
def interpret(request):
    return request.param


def backend(interpret, **kw):
    kw.setdefault("block_targets", 16)
    kw.setdefault("block_sources", 16)
    return PallasDirectSum(interpret=interpret, **kw)


@pytest.mark.parametrize("n", [1, 7, 16, 37])
def test_matches_bruteforce(interpret, n) -> None:
    check_brute(backend(interpret), *massive(n))


def test_tracer_targets_against_sources(interpret) -> None:
    s, _ = massive(21)
    rng = np.random.default_rng(1)
    t = Targets(
        q=jnp.asarray(rng.normal(size=(5, 3))),
        h=jnp.asarray(rng.uniform(0.01, 0.1, 5)),
        self_index=jnp.full(5, -1),
    )
    check_brute(backend(interpret), s, t)


def test_combine_override_changes_pair_softening(interpret) -> None:
    class MinKernel(PlummerKernel):
        def combine(self, h_i, h_j):
            return jnp.minimum(h_i, h_j)

    check_brute(backend(interpret, kernel=MinKernel()), *massive(19), np.minimum)


def test_newtons_third_law(interpret) -> None:
    s, t = massive(40)
    a = backend(interpret).accelerations(s, t, G=G)
    total = jnp.sum(s.m[:, None] * a, axis=0)
    scale = jnp.sum(s.m[:, None] * jnp.abs(a))
    np.testing.assert_allclose(total, 0.0, atol=1e-14 * float(scale))


@pytest.mark.parametrize("name", ["accelerations", "potential"])
def test_gradients_match_direct_sum(interpret, name) -> None:
    s, t = massive(37)

    def loss(b, s, t, G):
        return jnp.sum(jnp.sin(getattr(b, name)(s, t, G=G)))

    grad = jax.grad(loss, argnums=(1, 2, 3), allow_int=True)
    want = grad(DirectSum(chunk_size=8), s, t, G)
    got = grad(backend(interpret), s, t, G)
    for w, g in zip(jax.tree.leaves(want), jax.tree.leaves(got), strict=True):
        if jnp.issubdtype(w.dtype, jnp.inexact):
            np.testing.assert_allclose(g, w, rtol=1e-11, atol=1e-13)


def test_check_grads(interpret) -> None:
    s, t = massive(5)
    b = backend(interpret)
    check_grads(
        lambda q, m, h: b.accelerations(
            Sources(q, m, h), Targets(q, h, t.self_index), G=G
        ),
        (s.q, s.m, s.h),
        order=1,
        modes=("rev",),
    )


def test_coincident_particles_finite_value_and_grad(interpret) -> None:
    m, h, idx = jnp.ones(2), jnp.full(2, 0.05), jnp.arange(2)
    q0 = jnp.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    b = backend(interpret)

    def f(q):
        a = b.accelerations(Sources(q, m, h), Targets(q, h, idx), G=G)
        return jnp.sum(a**2) + jnp.sum(a)

    assert jnp.all(jnp.isfinite(jax.grad(f)(q0)))


def test_lone_particle_feels_no_self_potential(interpret) -> None:
    one = Sources(q=jnp.zeros((1, 3)), m=jnp.ones(1), h=jnp.zeros(1))
    p = backend(interpret).potential(one, Targets(one.q, one.h, jnp.array([0])), G=G)
    assert p[0] == 0.0


@pytest.mark.parametrize("name", ["accelerations", "potential"])
def test_non_finite_target_is_nan_and_isolated(interpret, name) -> None:
    src_q = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    b = backend(interpret)

    def run(src_q, x):
        s = Sources(q=src_q, m=jnp.ones(2), h=jnp.zeros(2))
        t = Targets(q=x, h=jnp.zeros(2), self_index=jnp.asarray([-1, -1]))
        return getattr(b, name)(s, t, G=1.0)

    def loss(src_q, x):
        return jnp.sum(run(src_q, x)[0] ** 2)

    good = jnp.asarray([[1.0, 1.0, 0.0], [3.0, 0.0, 0.0]])
    bad = good.at[1].set(jnp.nan)
    assert jnp.all(jnp.isnan(run(src_q, bad)[1]))
    np.testing.assert_allclose(
        jax.grad(loss)(src_q, bad), jax.grad(loss)(src_q, good), rtol=1e-13
    )


@pytest.mark.parametrize("policy", ["stop", "none"])
def test_grad_policies(interpret, policy) -> None:
    s, t = massive(9)
    b = backend(interpret, grad=policy)
    np.testing.assert_allclose(
        b.accelerations(s, t, G=G),
        DirectSum().accelerations(s, t, G=G),
        rtol=1e-12,
    )


def test_vmap_over_positions(interpret) -> None:
    s, t = massive(11)
    qs = jnp.stack([s.q, 2 * s.q])
    b = backend(interpret)

    def acc(q):
        return b.accelerations(Sources(q, s.m, s.h), Targets(q, t.h, t.self_index), G=G)

    want = jnp.stack(
        [
            DirectSum().accelerations(
                Sources(q, s.m, s.h), Targets(q, t.h, t.self_index), G=G
            )
            for q in qs
        ]
    )
    np.testing.assert_allclose(jax.vmap(acc)(qs), want, rtol=1e-12)


def test_drop_in_for_nbody_field(interpret) -> None:
    """NBodyField gives the same accelerations with either backend."""
    import galax.potential as gp

    s, _ = massive(13)
    pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
    fields = [
        gn.NBodyField(s.m * 1e9, eps=0.01, external_potential=pot, backend=b)
        for b in (DirectSum(), backend(interpret))
    ]
    want, got = (f.acceleration(0.0, s.q) for f in fields)
    np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize("name", ["accelerations", "potential"])
def test_lowers_for_tpu(name) -> None:
    """Pallas -> Mosaic lowering only; not run on TPU hardware."""
    from jax import export

    from galax.nbody._src.pallas_direct import (
        _Config,
        pallas_accelerations,
        pallas_potential,
    )

    fn = {"accelerations": pallas_accelerations, "potential": pallas_potential}[name]
    cfg = _Config(
        kernel=SplineKernel(),
        potential=False,
        block_targets=8,
        block_sources=128,
        num_warps=None,
        num_stages=None,
        interpret=False,
        precision="float32",
        compensated=True,
    )
    s, t = massive(20)  # float64 inputs, as galax uses

    def loss(q, m, h):
        out = fn(Sources(q, m, h), Targets(q, h, t.self_index), G=G, cfg=cfg)
        return jnp.sum(out)

    exp = export.export(jax.jit(jax.value_and_grad(loss, (0, 1, 2))), platforms=["tpu"])
    assert "tpu_custom_call" in exp(s.q, s.m, s.h).mlir_module()


@pytest.mark.parametrize(
    "kernel",
    [NewtonianKernel(), DehnenK1Kernel(), DehnenK2Kernel(), SplineKernel()],
    ids=lambda k: type(k).__name__,
)
def test_kernels_match_direct_sum(interpret, kernel) -> None:
    s, t = massive(37)
    s = Sources(s.q, s.m, 5 * s.h)  # put some pairs inside the spline support
    t = Targets(t.q, s.h, t.self_index)
    want, got = (
        DirectSum(kernel=kernel, chunk_size=8),
        backend(interpret, kernel=kernel),
    )
    for name in ("accelerations", "potential"):
        np.testing.assert_allclose(
            getattr(got, name)(s, t, G=G), getattr(want, name)(s, t, G=G), rtol=1e-11
        )
    loss = lambda b, q: jnp.sum(jnp.sin(b.accelerations(Sources(q, s.m, s.h), t, G=G)))
    np.testing.assert_allclose(
        jax.grad(loss, 1)(got, s.q), jax.grad(loss, 1)(want, s.q), rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("compensated", [False, True])
def test_float32_precision(interpret, compensated) -> None:
    """Float32 pair sum on far-off-centre float64 inputs; result stays float64."""
    s, t = massive(300)
    off = jnp.asarray([1e4, -2e4, 3e4])  # recentring keeps this from costing digits
    s, t = Sources(s.q + off, s.m, s.h), Targets(t.q + off, t.h, t.self_index)
    want = DirectSum().accelerations(s, t, G=G)
    b = backend(interpret, precision="float32", compensated=compensated)
    got = b.accelerations(s, t, G=G)
    assert got.dtype == want.dtype
    err = jnp.max(jnp.linalg.norm(got - want, axis=1) / jnp.linalg.norm(want, axis=1))
    assert err < 1e-4
    g = jax.grad(lambda q: jnp.sum(b.accelerations(Sources(q, s.m, s.h), t, G=G)))(s.q)
    assert g.dtype == s.q.dtype
    assert jnp.all(jnp.isfinite(g))


def test_kahan_reduces_error(interpret) -> None:
    """Summing many small terms in float32: Kahan beats plain summation."""
    n = 4096
    rng = np.random.default_rng(3)
    q = jnp.asarray(rng.normal(size=(n, 3)) * 10)
    s = Sources(q, jnp.ones(n), jnp.full(n, 0.1))
    t = Targets(q[:16], s.h[:16], jnp.arange(16))
    want = DirectSum().potential(s, t, G=G)

    def err(*, c):
        b = backend(interpret, precision="float32", compensated=c, block_sources=16)
        return float(jnp.max(jnp.abs(b.potential(s, t, G=G) / want - 1)))

    assert err(c=True) <= err(c=False)

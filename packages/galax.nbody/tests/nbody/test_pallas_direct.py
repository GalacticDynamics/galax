"""Tests for the Pallas direct-sum backend.

Run in interpret mode everywhere, and compiled too when a GPU is present.
"""

import functools as ft

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import galax.nbody as gn
from .test_direct import G, check_brute, massive
from galax.nbody._src.direct import DirectSum
from galax.nbody._src.kernels import (
    DehnenK1Kernel,
    DehnenK2Kernel,
    NewtonianKernel,
    PlummerKernel,
    SplineKernel,
)
from galax.nbody._src.pallas_direct import PallasDirectSum
from galax.nbody._src.particles import Sources, Targets

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


@pytest.mark.parametrize("name", ["accelerations", "potential"])
def test_check_grads(interpret, name) -> None:
    """Forward and reverse mode up to second order, against finite differences."""
    s, t = massive(5)
    b = backend(interpret)
    check_grads(
        lambda q, m, h: getattr(b, name)(
            Sources(q, m, h), Targets(q, h, t.self_index), G=G
        ),
        (s.q, s.m, s.h),
        order=2,
        modes=("fwd", "rev"),
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
    # Forward mode end to end, as variational equations use it.
    want, got = (jax.jacfwd(lambda q, f=f: f.acceleration(0.0, q))(s.q) for f in fields)
    np.testing.assert_allclose(got, want, rtol=1e-10)


@pytest.mark.skipif(
    jax.default_backend() in ("gpu", "tpu"), reason="fallback is CPU-only"
)
@pytest.mark.parametrize("name", ["accelerations", "potential"])
def test_cpu_fallback_matches_direct_sum(name) -> None:
    """Compiled mode on CPU falls back to DirectSum, value and forward mode."""
    s, t = massive(17)
    f, fd = (
        _fns(b, name, t)[0]
        for b in (
            backend(interpret=False),
            DirectSum(chunk_size=PallasDirectSum.fallback_chunk_size),
        )
    )
    args = (s.q, s.m, s.h)
    np.testing.assert_array_equal(f(*args), fd(*args))
    tan = _tangents(args)
    _close(jax.jvp(f, args, tan), jax.jvp(fd, args, tan), 1e-14)


def test_lowers_for_gpu_with_launch_params() -> None:
    """Pallas -> Triton lowering passes num_warps/num_stages; runs on CPU CI."""
    from jax import export

    from galax.nbody._src.pallas_direct import _Config, pallas_accelerations

    cfg = _Config(
        kernel=PlummerKernel(),
        potential=False,
        block_targets=16,
        block_sources=64,
        num_warps=4,
        num_stages=2,
        interpret=False,
    )
    s, t = massive(20)

    def acc(q):
        return pallas_accelerations(
            Sources(q, s.m, s.h), Targets(q, s.h, t.self_index), G=G, cfg=cfg
        )

    triton = "__gpu$xla.gpu.triton"
    exp = export.export(
        jax.jit(jax.jvp, static_argnums=0),
        platforms=["cuda"],
        disabled_checks=[export.DisabledSafetyCheck.custom_call(triton)],
    )(acc, (s.q,), (s.q,))
    mlir = exp.mlir_module()
    assert triton in mlir
    assert "num_warps = 4" in mlir
    assert "num_stages = 2" in mlir


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


def test_tpu_defaults_to_float32(monkeypatch) -> None:
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    assert PallasDirectSum()._cfg().precision == "float32"
    assert PallasDirectSum(precision="bfloat16")._cfg().precision == "bfloat16"
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert PallasDirectSum()._cfg().precision is None


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


# --- Full autodiff ------------------------------------------------------------


def _fns(b, name, t):
    """Value and scalar loss of ``(q, m, h)`` for sources = targets."""

    def f(q, m, h):
        return getattr(b, name)(Sources(q, m, h), Targets(q, h, t.self_index), G=G)

    return f, lambda q, m, h: jnp.sum(jnp.sin(f(q, m, h)))


def _close(got, want, rtol=1e-10) -> None:
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(g, w, rtol=rtol, atol=rtol * float(jnp.max(abs(w))))


def _tangents(args, seed=2):
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.normal(size=x.shape)) for x in args)


@pytest.mark.parametrize("slices", [2, 3, 8])
@pytest.mark.parametrize("name", ["accelerations", "potential"])
def test_source_slices_match_direct_sum(interpret, name, slices) -> None:
    """Split source loops (4 full tiles + a padded one; 8 > tiles is allowed)."""
    s, t = massive(37)
    args = (s.q, s.m, s.h)
    b = backend(interpret, block_targets=8, block_sources=8, source_slices=slices)
    fp, lp = _fns(b, name, t)
    fd, ld = _fns(DirectSum(), name, t)
    _close(fp(*args), fd(*args))
    _close(jax.grad(lp, (0, 1, 2))(*args), jax.grad(ld, (0, 1, 2))(*args))
    tan = _tangents(args)
    _close(jax.jvp(fp, args, tan), jax.jvp(fd, args, tan))


def test_source_slices_auto_rule(monkeypatch) -> None:
    """Split small grids on GPU only; a vmapped batch counts toward the grid."""
    from galax.nbody._src import pallas_direct as pd

    monkeypatch.setattr(pd, "_sm_count", lambda: 100)
    lc = pd._Launch(16, 64, 4, None, interpret=False)

    def slices(lead, n, lc=lc, *, gpu=True):
        avals = [jax.ShapeDtypeStruct((lead, n), jnp.float32)] * 11
        return pd._slices(lc, avals, 6, gpu=gpu)

    assert slices(1, 4096) == pd._SPLIT_MAX  # 256 programs, 64 tiles
    assert slices(1, 1024) == 2  # 16 tiles: at least 8 per slice
    assert slices(1, 1 << 16) == 1  # grid already full
    assert slices(64, 4096) == 1  # vmapped batch fills it
    assert slices(1, 4096, lc._replace(slices=3)) == 3  # explicit
    assert slices(1, 4096, lc._replace(slices=3), gpu=False) == 1  # TPU


@pytest.mark.parametrize("name", ["accelerations", "potential"])
def test_forward_mode_matches_direct_sum(interpret, name) -> None:
    s, t = massive(13)
    args, tan = (s.q, s.m, s.h), _tangents((s.q, s.m, s.h))
    fp, _ = _fns(backend(interpret, block_targets=8, block_sources=8), name, t)
    fd, _ = _fns(DirectSum(chunk_size=4), name, t)
    _close(jax.jvp(fp, args, tan), jax.jvp(fd, args, tan))
    _close(jax.linearize(fp, *args)[1](*tan), jax.linearize(fd, *args)[1](*tan))
    _close(jax.jacfwd(fp, (0, 1, 2))(*args), jax.jacrev(fd, (0, 1, 2))(*args))


@pytest.mark.parametrize("name", ["accelerations", "potential"])
def test_second_order_matches_direct_sum(interpret, name) -> None:
    s, t = massive(9)
    args = (s.q, s.m, s.h)
    _, lp = _fns(backend(interpret, block_targets=8, block_sources=8), name, t)
    _, ld = _fns(DirectSum(chunk_size=4), name, t)
    want = jax.hessian(ld, (0, 1, 2))(*args)
    _close(jax.hessian(lp, (0, 1, 2))(*args), want, 1e-9)  # fwd-over-rev
    rev2 = jax.jacrev(jax.grad(lp, (0, 1, 2)), (0, 1, 2))  # rev-over-rev
    _close(rev2(*args), want, 1e-9)
    fwd2 = jax.jacfwd(jax.jacfwd(lp, (0, 1, 2)), (0, 1, 2))  # fwd-over-fwd
    _close(fwd2(*args), want, 1e-9)


def test_third_order_matches_direct_sum(interpret) -> None:
    s, t = massive(7)

    def g3(loss):
        def inner(q):
            return jnp.sum(jax.grad(loss)(q, s.m, s.h) ** 2)

        return jax.jvp(
            jax.grad(lambda q: jnp.sum(jax.grad(inner)(q) ** 2)),
            (s.q,),
            (jnp.ones_like(s.q),),
        )

    _, lp = _fns(backend(interpret), "accelerations", t)
    _, ld = _fns(DirectSum(), "accelerations", t)
    _close(g3(lp), g3(ld), 1e-8)


@pytest.mark.parametrize(
    "kernel",
    [NewtonianKernel(), DehnenK1Kernel(), DehnenK2Kernel(), SplineKernel()],
    ids=lambda k: type(k).__name__,
)
def test_kernels_second_order(interpret, kernel) -> None:
    s, t = massive(9)
    s = Sources(s.q, s.m, 5 * s.h)  # put some pairs inside the spline support
    t = Targets(t.q, s.h, t.self_index)
    args = (s.q, s.m, s.h)
    for name in ("accelerations", "potential"):
        _, lp = _fns(backend(interpret, kernel=kernel, block_targets=8), name, t)
        _, ld = _fns(DirectSum(kernel=kernel), name, t)
        _close(jax.hessian(lp, (0, 2))(*args), jax.hessian(ld, (0, 2))(*args), 1e-8)


def test_vmap_compositions(interpret) -> None:
    s, t = massive(11)
    qs = jnp.stack([s.q, 1.1 * s.q, 0.9 * s.q])
    fp, lp = _fns(
        backend(interpret, block_targets=8, block_sources=8), "accelerations", t
    )
    fd, ld = _fns(DirectSum(), "accelerations", t)
    vgrad = lambda l: jax.vmap(jax.grad(l), (0, None, None))(qs, s.m, s.h)
    _close(vgrad(lp), vgrad(ld))

    def gvmap(f):
        def loss(q):
            return jnp.sum(jnp.cos(jax.vmap(f, (0, None, None))(q * qs, s.m, s.h)))

        return jax.grad(loss)(s.q)

    _close(gvmap(fp), gvmap(fd))
    tan = _tangents((s.q,))[0]

    def vjvp(f):
        return jax.vmap(lambda v: jax.jvp(lambda q: f(q, s.m, s.h), (s.q,), (v,))[1])(
            jnp.stack([tan, -2 * tan])
        )

    _close(vjvp(fp), vjvp(fd))
    vv = lambda f: jax.vmap(jax.vmap(f, (0, None, None)), (None, 0, None))
    ms = jnp.stack([s.m, 2 * s.m])
    _close(vv(fp)(qs, ms, s.h), vv(fd)(qs, ms, s.h))


def test_vmap_over_targets_shares_sources(interpret) -> None:
    """Only the targets are batched; the sources stay shared."""
    s, _ = massive(21)
    rng = np.random.default_rng(4)
    tq = jnp.asarray(rng.normal(size=(3, 5, 3)))
    th, ti = jnp.full(5, 0.05), jnp.full(5, -1)
    b = backend(interpret, block_targets=8, block_sources=8)

    def f(backend, m):
        def one(q):
            return backend.accelerations(Sources(s.q, m, s.h), Targets(q, th, ti), G=G)

        return jnp.sum(jax.vmap(one)(tq) ** 2)

    _close(f(b, s.m), f(DirectSum(), s.m))
    _close(jax.grad(f, 1)(b, s.m), jax.grad(f, 1)(DirectSum(), s.m))


def test_shard_map_targets_sharded(interpret) -> None:
    """Targets split over a mesh, sources gathered; `check_vma` stays on."""
    from jax.sharding import Mesh, PartitionSpec as P

    mesh = Mesh(np.array(jax.devices()), ("d",))
    s, _ = massive(8 * mesh.size)

    def sharded(backend):
        def body(q, m, h):
            idx = jax.lax.axis_index("d") * q.shape[0] + jnp.arange(q.shape[0])
            g = lambda x: jax.lax.all_gather(x, "d", tiled=True)
            src = Sources(g(q), g(m), g(h))
            return backend.accelerations(src, Targets(q, h, idx), G=G)

        return jax.shard_map(body, mesh=mesh, in_specs=P("d"), out_specs=P("d"))

    fp = sharded(backend(interpret, block_targets=8, block_sources=8))
    fd = sharded(DirectSum())
    args = (s.q, s.m, s.h)
    _close(fp(*args), fd(*args))
    loss = lambda f: lambda *a: jnp.sum(jnp.sin(f(*a)))
    _close(jax.grad(loss(fp), (0, 1))(*args), jax.grad(loss(fd), (0, 1))(*args))
    tan = _tangents((s.q,))[0]
    jv = lambda f: jax.jvp(lambda q: f(q, s.m, s.h), (s.q,), (tan,))[1]
    _close(jv(fp), jv(fd))

    # Independent systems: the batch is split over the mesh, vmapped inside.
    def batched(backend):
        def one(q, m, h):
            t = Targets(q, h, jnp.arange(q.shape[0]))
            return backend.accelerations(Sources(q, m, h), t, G=G)

        return jax.shard_map(
            jax.vmap(one), mesh=mesh, in_specs=P("d"), out_specs=P("d")
        )

    k = 2 * mesh.size
    bargs = (
        s.q * jnp.linspace(0.5, 1.5, k)[:, None, None],
        jnp.broadcast_to(s.m, (k, *s.m.shape)),
        jnp.broadcast_to(s.h, (k, *s.h.shape)),
    )
    _close(batched(backend(interpret))(*bargs), batched(DirectSum())(*bargs))


@pytest.mark.parametrize("name", ["accelerations", "potential"])
def test_non_finite_target_isolated_in_forward_mode(interpret, name) -> None:
    src_q = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    b = backend(interpret)

    def run(src_q, x):
        s = Sources(q=src_q, m=jnp.ones(2), h=jnp.zeros(2))
        t = Targets(q=x, h=jnp.zeros(2), self_index=jnp.asarray([-1, -1]))
        return getattr(b, name)(s, t, G=1.0)[0]

    good = jnp.asarray([[1.0, 1.0, 0.0], [3.0, 0.0, 0.0]])
    bad = good.at[1].set(jnp.nan)
    hess = jax.hessian(lambda q, x: jnp.sum(run(q, x) ** 2))
    np.testing.assert_allclose(hess(src_q, bad), hess(src_q, good), rtol=1e-13)
    np.testing.assert_allclose(
        jax.jacfwd(run)(src_q, bad), jax.jacfwd(run)(src_q, good), rtol=1e-13
    )


def test_coincident_particles_finite_second_order(interpret) -> None:
    m, h, idx = jnp.ones(2), jnp.full(2, 0.05), jnp.arange(2)
    q0 = jnp.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    b = backend(interpret)

    def f(q):
        return jnp.sum(b.accelerations(Sources(q, m, h), Targets(q, h, idx), G=G) ** 2)

    assert jnp.all(jnp.isfinite(jax.hessian(f)(q0)))


def test_float32_precision_forward_mode(interpret) -> None:
    s, t = massive(40)
    b = backend(interpret, precision="float32")
    want = jax.jvp(
        _fns(DirectSum(), "accelerations", t)[0],
        (s.q, s.m, s.h),
        _tangents((s.q, s.m, s.h)),
    )
    got = jax.jvp(
        _fns(b, "accelerations", t)[0], (s.q, s.m, s.h), _tangents((s.q, s.m, s.h))
    )
    assert all(g.dtype == jnp.float64 for g in got)
    _close(got, want, 1e-4)


def test_hessian_lowers_for_tpu() -> None:
    """Second-order kernels lower through Mosaic; not run on TPU hardware."""
    from jax import export

    from galax.nbody._src.pallas_direct import _Config, pallas_potential

    cfg = _Config(
        kernel=PlummerKernel(),
        potential=True,
        block_targets=8,
        block_sources=128,
        num_warps=None,
        num_stages=None,
        interpret=False,
        precision="float32",
    )
    s, t = massive(20)

    def loss(q):
        return jnp.sum(
            pallas_potential(
                Sources(q, s.m, s.h), Targets(q, s.h, t.self_index), G=G, cfg=cfg
            )
        )

    exp = export.export(jax.jit(jax.hessian(loss)), platforms=["tpu"])
    assert "tpu_custom_call" in exp(s.q).mlir_module()


# ---------------------------------------------------------------------------
# Fused two-sided VJP (`pair_sum_ab_p`). Compiled on GPU it runs the atomic
# kernel; elsewhere it lowers to the two pair sums, so these check its rules.


def _fused_pair(*, interpret, fused, name="accelerations"):
    b = backend(interpret, fused_grad=fused)
    t = massive(23)[1]
    return _fns(b, name, t)


@pytest.mark.parametrize("name", ["accelerations", "potential"])
def test_fused_grad_matches_unfused(interpret, name) -> None:
    s, _ = massive(23)
    args = (s.q, s.m, s.h)
    (_, lf), (_, lu) = (
        _fused_pair(interpret=interpret, fused=f, name=name) for f in (True, False)
    )
    _close(jax.grad(lf, (0, 1, 2))(*args), jax.grad(lu, (0, 1, 2))(*args))
    _close(jax.hessian(lf, (0, 1, 2))(*args), jax.hessian(lu, (0, 1, 2))(*args), 1e-9)
    rr = lambda f: jax.jacrev(jax.grad(f))(*args)  # transposes the fused op's JVP
    _close(rr(lf), rr(lu), 1e-9)
    vg = lambda f: jax.vmap(jax.grad(f), (0, None, None))(
        jnp.stack([s.q, 1.1 * s.q]), s.m, s.h
    )
    _close(vg(lf), vg(lu))


def test_fused_grad_policy() -> None:
    s, _ = massive(9)
    args = (s.q, s.m, s.h)

    def used(fused, dtype):
        _, loss = _fused_pair(interpret=True, fused=fused)
        jpr = jax.make_jaxpr(jax.grad(loss))(*(x.astype(dtype) for x in args))
        return "nbody_pair_sum_ab" in str(jpr)

    assert used(None, jnp.float64)
    assert not used(None, jnp.float32)
    assert used(True, jnp.float32)  # noqa: FBT003
    assert not used(False, jnp.float64)  # noqa: FBT003


def test_fused_linear_transpose() -> None:
    """The fused op is linear in the cotangent; transposing it uses its rule."""
    s, t = massive(11)
    f, _ = _fused_pair(interpret=True, fused=True)
    _, vjp_f = jax.vjp(lambda q: f(q, s.m, s.h), s.q)
    _, vjp_u = jax.vjp(
        lambda q: _fused_pair(interpret=True, fused=False)[0](q, s.m, s.h), s.q
    )
    ct = jnp.ones_like(s.q)
    tf = jax.linear_transpose(lambda c: vjp_f(c)[0], ct)
    tu = jax.linear_transpose(lambda c: vjp_u(c)[0], ct)
    _close(tf(s.q), tu(s.q))


@pytest.mark.parametrize("n", [5, 37])
def test_fused_kernel_matches_two_pass(n) -> None:
    """The atomic kernel itself, in interpret mode (padding on both sides)."""
    from galax.nbody._src.pallas_direct import (
        _Base,
        _Both,
        _call_ab,
        _Launch,
        _Vjp,
        pair_sum,
    )

    s, t = massive(n)
    rng = np.random.default_rng(3)
    a = (*s.q.T, s.h, t.self_index.astype(jnp.int32), jnp.ones(n, jnp.int32))
    b = (*s.q.T, s.m, s.h)
    ct = tuple(jnp.asarray(rng.normal(size=n)) for _ in range(3))
    da, db = tuple(x.dtype for x in a), tuple(x.dtype for x in b)
    f, wrt = _Base(NewtonianKernel(), potential=False), (0, 1, 2, 3)
    lc = _Launch(16, 16, None, None, interpret=True, split=True)
    kw = {"n_ct": 3, "dtypes_a": da, "dtypes_b": db}
    row = lambda xs: [x[None] for x in xs]
    got = _call_ab(
        *row((*a, *ct, *b)), fn=_Both(f, wrt, wrt, **kw), n_a=len(a) + 3, lc=lc
    )
    want_a = pair_sum(_Vjp(f, "a", wrt, **kw), row((*a, *ct)), row(b), lc, n_lin_a=3)
    want_b = pair_sum(_Vjp(f, "b", wrt, **kw), row(b), row((*a, *ct)), lc, n_lin_b=3)
    _close(got, [*want_a, *want_b], 1e-12)


def test_fused_lowers_for_gpu() -> None:
    """The f64 gradient lowers to the fused Triton kernel; runs on CPU CI."""
    from jax import export

    from galax.nbody._src.pallas_direct import _Config, pallas_accelerations

    cfg = _Config(
        kernel=PlummerKernel(),
        potential=False,
        block_targets=16,
        block_sources=64,
        num_warps=4,
        num_stages=None,
        interpret=False,
    )
    s, t = massive(20)

    def loss(q):
        acc = pallas_accelerations(
            Sources(q, s.m, s.h), Targets(q, s.h, t.self_index), G=G, cfg=cfg
        )
        return jnp.sum(jnp.sin(acc))

    triton = "__gpu$xla.gpu.triton"
    exp = export.export(
        jax.jit(jax.grad(loss)),
        platforms=["cuda"],
        disabled_checks=[export.DisabledSafetyCheck.custom_call(triton)],
    )(s.q)
    assert "nbody_pair_sum_fused" in exp.mlir_module()


def test_self_pairs_in_any_tile(interpret) -> None:
    """Self pairs off the diagonal tiles, mixed with tracers and a NaN target."""
    s, _ = massive(53)
    rng = np.random.default_rng(5)
    perm = rng.permutation(53)
    si = jnp.asarray(np.where(rng.uniform(size=53) < 0.3, -1, perm))
    q = s.q[perm].at[7].set(jnp.nan)
    t = Targets(q=q, h=s.h[perm], self_index=si)
    b = backend(interpret, block_targets=8, block_sources=8)
    ok = np.isfinite(np.asarray(q)).all(1)
    tt = Targets(q=q[ok], h=t.h[ok], self_index=si[ok])
    check_brute(b, s, tt)
    acc = b.accelerations(s, t, G=G)
    assert np.isnan(acc[7]).all()
    assert np.isfinite(np.delete(acc, 7, 0)).all()

    def loss(bk, q):
        out = bk.accelerations(Sources(q, s.m, s.h), t, G=G)
        return jnp.sum(jnp.sin(jnp.where(jnp.isfinite(out), out, 0.0)))

    _close(
        jax.grad(ft.partial(loss, b))(s.q), jax.grad(ft.partial(loss, DirectSum()))(s.q)
    )

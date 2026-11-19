"""Direct-sum force backend as a tiled Pallas kernel.

Each program owns ``block_targets`` targets and loops over sources in tiles of
``block_sources``, so the pair matrix is never stored. Reverse mode is a
`jax.custom_vjp` whose two backward kernels differentiate the pair terms with
`jax.vjp` in-kernel. Padding, NaN targets and ``G`` are handled in plain JAX.

Tested on NVIDIA GPUs (Ampere, Ada). TPU lowers but is untested on hardware.
"""

__all__ = ["PallasDirectSum", "pallas_accelerations", "pallas_potential"]

import functools as ft

from jaxtyping import Array
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .backend import AbstractForceBackend
from .direct import direct_sum_accelerations, direct_sum_potential
from .kernels import AbstractSofteningKernel, PlummerKernel
from .particles import Sources, Targets

_Arrs = tuple[Array, ...]


def _sds(shape: tuple[int, int], dtype: Any) -> Any:
    return jax.ShapeDtypeStruct(shape, dtype)  # type: ignore[no-untyped-call]


class _Config(NamedTuple):
    """Static launch configuration."""

    kernel: AbstractSofteningKernel
    potential: bool
    block_targets: int
    block_sources: int
    num_warps: int | None
    num_stages: int | None
    interpret: bool
    precision: str | None = None
    compensated: bool = False


def _tile(
    cfg: _Config,
    t: tuple[Array, ...],
    s: tuple[Array, ...],
    it: Array,
    okt: Array,
    j0: Array,
    *,
    reduce: bool = True,
) -> tuple[Array, ...]:
    """Return one tile's pair terms, summed over sources if ``reduce``.

    ``t = (x, y, z, h)`` are (BT, 1) columns, ``s = (x, y, z, m, h)`` are
    (1, BS) rows, ``j0`` is the first source index. Pairs with ``it == j`` or
    ``okt == 0`` are skipped.
    """
    xt, yt, zt, ht = t
    xs, ys, zs, ms, hs = s
    dx, dy, dz = xs - xt, ys - yt, zs - zt
    r2 = dx * dx + dy * dy + dz * dz
    js = j0 + jax.lax.broadcasted_iota(jnp.int32, xs.shape, 1)
    skip = (it == js) | (okt == 0)
    # Double where, as in DirectSum: finite value and gradient when skipped.
    r2s = jnp.where(skip, jnp.ones_like(r2), r2)
    hp = cfg.kernel.combine(ht, hs)
    if cfg.potential:
        pot = cfg.kernel.potential(r2s, hp)
        w = jnp.where(skip, jnp.zeros_like(pot), pot) * ms
        terms: tuple[Array, ...] = (w,)
    else:
        f = cfg.kernel.force_factor(r2s, hp)
        w = jnp.where(skip, jnp.zeros_like(f), f) * ms
        terms = (w * dx, w * dy, w * dz)
    if not reduce:
        return terms
    return tuple(jnp.sum(x, axis=1, keepdims=True) for x in terms)


def _n_out(cfg: _Config) -> int:
    return 1 if cfg.potential else 3


def _compiler_params(cfg: _Config) -> Any:
    if cfg.interpret or jax.default_backend() != "gpu":
        return None
    from jax.experimental.pallas import triton as pltr

    return pltr.CompilerParams(num_warps=cfg.num_warps, num_stages=cfg.num_stages)


def _i32(x: int) -> Array:
    """Loop bounds as int32; Mosaic (TPU) can't lower int64 under x64."""
    return jnp.asarray(x, dtype=jnp.int32)


# Kernel refs: 4 target columns, it, okt, 5 source rows, [cotangents], outputs.


def _fwd_kernel(*refs: Any, cfg: _Config, n_src: int) -> None:
    t_refs, it_ref, okt_ref, s_refs = refs[:4], refs[4], refs[5], refs[6:11]
    out_refs = refs[11:]
    t = tuple(r[...] for r in t_refs)
    it, okt = it_ref[...], okt_ref[...]
    bs = cfg.block_sources

    def body(k: Array, carry: tuple[_Arrs, _Arrs]) -> tuple[_Arrs, _Arrs]:
        acc, comp = carry
        j0 = k * bs
        s = tuple(r[:, pl.ds(j0, bs)] for r in s_refs)
        terms = _tile(cfg, t, s, it, okt, j0, reduce=False)
        if not cfg.compensated:
            return tuple(a + p for a, p in zip(acc, terms, strict=True)), comp
        ys = tuple(p - c for p, c in zip(terms, comp, strict=True))  # Kahan
        new = tuple(a + y for a, y in zip(acc, ys, strict=True))
        comp = tuple((n - a) - y for n, a, y in zip(new, acc, ys, strict=True))
        return new, comp

    # Accumulate per pair; reduce across lanes once at the end.
    zero = (jnp.zeros((cfg.block_targets, bs), t[0].dtype),) * _n_out(cfg)
    carry = (zero, zero if cfg.compensated else ())
    acc, comp = jax.lax.fori_loop(_i32(0), _i32(n_src // bs), body, carry)
    if cfg.compensated:
        acc = tuple(a - c for a, c in zip(acc, comp, strict=True))
    for o, a in zip(out_refs, acc, strict=True):
        o[...] = jnp.sum(a, axis=1, keepdims=True)


def _bwd_targets_kernel(*refs: Any, cfg: _Config, n_src: int) -> None:
    """Target gradients, looping over source tiles."""
    t_refs, it_ref, okt_ref, s_refs = refs[:4], refs[4], refs[5], refs[6:11]
    n = _n_out(cfg)
    g = tuple(r[...] for r in refs[11 : 11 + n])
    out_refs = refs[11 + n :]
    t = tuple(r[...] for r in t_refs)
    it, okt = it_ref[...], okt_ref[...]
    bs = cfg.block_sources

    shape = (cfg.block_targets, bs)
    # Differentiate w.r.t. broadcast inputs to get per-pair gradients.
    tb = tuple(jnp.broadcast_to(x, shape) for x in t)
    gb = tuple(jnp.broadcast_to(x, shape) for x in g)

    def body(k: Array, acc: tuple[Array, ...]) -> tuple[Array, ...]:
        j0 = k * bs
        s = tuple(r[:, pl.ds(j0, bs)] for r in s_refs)
        _, vjp = jax.vjp(lambda t_: _tile(cfg, t_, s, it, okt, j0, reduce=False), tb)
        (gt,) = vjp(gb)
        return tuple(a + b for a, b in zip(acc, gt, strict=True))

    zero = jnp.zeros(shape, t[0].dtype)
    acc = jax.lax.fori_loop(_i32(0), _i32(n_src // bs), body, (zero,) * 4)
    for o, a in zip(out_refs, acc, strict=True):
        o[...] = jnp.sum(a, axis=1, keepdims=True)


def _bwd_sources_kernel(*refs: Any, cfg: _Config, n_tgt: int) -> None:
    """Source gradients, looping over target tiles."""
    t_refs, it_ref, okt_ref, s_refs = refs[:4], refs[4], refs[5], refs[6:11]
    n = _n_out(cfg)
    g_refs = refs[11 : 11 + n]
    out_refs = refs[11 + n :]
    bt = cfg.block_targets
    shape = (bt, cfg.block_sources)
    sb = tuple(jnp.broadcast_to(r[...], shape) for r in s_refs)
    j0 = pl.program_id(0) * cfg.block_sources

    def body(k: Array, acc: tuple[Array, ...]) -> tuple[Array, ...]:
        rows = pl.ds(k * bt, bt)
        t = tuple(r[rows, :] for r in t_refs)
        it, okt = it_ref[rows, :], okt_ref[rows, :]
        g = tuple(jnp.broadcast_to(r[rows, :], shape) for r in g_refs)
        _, vjp = jax.vjp(lambda s_: _tile(cfg, t, s_, it, okt, j0, reduce=False), sb)
        (gs,) = vjp(g)
        return tuple(a + b for a, b in zip(acc, gs, strict=True))

    zero = jnp.zeros(shape, sb[0].dtype)
    acc = jax.lax.fori_loop(_i32(0), _i32(n_tgt // bt), body, (zero,) * 5)
    for o, a in zip(out_refs, acc, strict=True):
        o[...] = jnp.sum(a, axis=0, keepdims=True)


def _specs(cfg: _Config, nt: int, ns: int, *, over: str) -> tuple[Any, Any]:
    """Return the (target column, source row) BlockSpecs for a 1D grid."""
    bt, bs = cfg.block_targets, cfg.block_sources
    if over == "targets":
        col = pl.BlockSpec((bt, 1), lambda i: (i, 0))
        row = pl.BlockSpec((1, ns), lambda _: (0, 0))
    else:
        col = pl.BlockSpec((nt, 1), lambda _: (0, 0))
        row = pl.BlockSpec((1, bs), lambda i: (0, i))
    return col, row


def _call_fwd(cfg: _Config, t: _Arrs, it: Array, okt: Array, s: _Arrs) -> _Arrs:
    nt, ns = t[0].shape[0], s[0].shape[1]
    col, row = _specs(cfg, nt, ns, over="targets")
    out = _sds((nt, 1), t[0].dtype)
    res: list[Array] = pl.pallas_call(
        ft.partial(_fwd_kernel, cfg=cfg, n_src=ns),
        grid=(nt // cfg.block_targets,),
        in_specs=[col] * 6 + [row] * 5,
        out_specs=[col] * _n_out(cfg),
        out_shape=[out] * _n_out(cfg),
        compiler_params=_compiler_params(cfg),
        interpret=cfg.interpret,
        name="nbody_direct_fwd",
    )(*t, it, okt, *s)
    return tuple(res)


def _call_bwd(
    cfg: _Config, t: _Arrs, it: Array, okt: Array, s: _Arrs, g: _Arrs
) -> tuple[_Arrs, _Arrs]:
    nt, ns = t[0].shape[0], s[0].shape[1]
    n = _n_out(cfg)
    params = _compiler_params(cfg)

    col, row = _specs(cfg, nt, ns, over="targets")
    gt = pl.pallas_call(
        ft.partial(_bwd_targets_kernel, cfg=cfg, n_src=ns),
        grid=(nt // cfg.block_targets,),
        in_specs=[col] * 6 + [row] * 5 + [col] * n,
        out_specs=[col] * 4,
        out_shape=[_sds((nt, 1), x.dtype) for x in t],
        compiler_params=params,
        interpret=cfg.interpret,
        name="nbody_direct_bwd_targets",
    )(*t, it, okt, *s, *g)

    col, row = _specs(cfg, nt, ns, over="sources")
    gs = pl.pallas_call(
        ft.partial(_bwd_sources_kernel, cfg=cfg, n_tgt=nt),
        grid=(ns // cfg.block_sources,),
        in_specs=[col] * 6 + [row] * 5 + [col] * n,
        out_specs=[row] * 5,
        out_shape=[_sds((1, ns), x.dtype) for x in s],
        compiler_params=params,
        interpret=cfg.interpret,
        name="nbody_direct_bwd_sources",
    )(*t, it, okt, *s, *g)
    return tuple(gt), tuple(gs)


@ft.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _core(cfg: _Config, t: _Arrs, it: Array, okt: Array, s: _Arrs) -> _Arrs:
    return tuple(_call_fwd(cfg, t, it, okt, s))


def _core_fwd(cfg: _Config, t: _Arrs, it: Array, okt: Array, s: _Arrs) -> Any:
    return _core(cfg, t, it, okt, s), (t, it, okt, s)


def _core_bwd(cfg: _Config, res: Any, g: _Arrs) -> Any:
    t, it, okt, s = res
    gt, gs = _call_bwd(cfg, t, it, okt, s, g)
    return gt, None, None, gs


_core.defvjp(_core_fwd, _core_bwd)


def _pad_to(n: int, b: int) -> int:
    return -(-n // b) * b


def _run(
    sources: Sources,
    targets: Targets,
    cfg: _Config,
) -> tuple[Array, Array]:
    """Pad and split into columns/rows, then run the kernel."""
    nt, ns = targets.q.shape[0], sources.q.shape[0]
    nt_p = _pad_to(nt, cfg.block_targets)
    ns_p = _pad_to(ns, cfg.block_sources)
    out_dtype = jnp.result_type(targets.q, sources.q, sources.m, sources.h, targets.h)
    dtype = out_dtype if cfg.precision is None else jnp.dtype(cfg.precision)

    ok = jnp.all(jnp.isfinite(targets.q), axis=-1)
    qt, qs = jnp.where(ok[:, None], targets.q, 0.0), sources.q
    if dtype != out_dtype:
        # Recentre on the source centre of mass before downcasting. Inactive
        # (parked, massless) rows don't move it.
        mtot = jnp.sum(sources.m)
        com = jnp.sum(sources.m[:, None] * qs, 0) / jnp.where(mtot == 0, 1, mtot)
        com = jax.lax.stop_gradient(jnp.where(mtot == 0, 0.0, com))
        qt, qs = qt - com, qs - com
    qt = qt.astype(dtype)

    def col(x: Array, fill: float = 0.0) -> Array:
        return jnp.pad(x, (0, nt_p - nt), constant_values=fill)[:, None]

    def row(x: Array, fill: float = 0.0) -> Array:
        return jnp.pad(x, (0, ns_p - ns), constant_values=fill)[None, :]

    # Padded sources: m = 0, h = 1 (finite zero terms). Padded targets: okt = 0.
    t = (col(qt[:, 0]), col(qt[:, 1]), col(qt[:, 2]), col(targets.h.astype(dtype)))
    it = col(targets.self_index.astype(jnp.int32), -1)
    okt = col(ok.astype(jnp.int32))
    q = qs.astype(dtype)
    s = (
        row(q[:, 0]),
        row(q[:, 1]),
        row(q[:, 2]),
        row(sources.m.astype(dtype)),
        row(sources.h.astype(dtype), 1.0),
    )
    out = _core(cfg, t, it, okt, s)
    return jnp.concatenate(out, axis=1)[:nt].astype(out_dtype), ok


def pallas_accelerations(
    sources: Sources, targets: Targets, *, G: float | Array, cfg: _Config
) -> Array:
    """Return the accelerations (Nt, 3) of ``targets`` from ``sources``."""
    acc, ok = _run(sources, targets, cfg._replace(potential=False))
    return jnp.where(ok[:, None], G * acc, jnp.nan)


def pallas_potential(
    sources: Sources, targets: Targets, *, G: float | Array, cfg: _Config
) -> Array:
    """Return the potentials (Nt,) of ``targets`` from ``sources``."""
    pot, ok = _run(sources, targets, cfg._replace(potential=True))
    return jnp.where(ok, G * pot[:, 0], jnp.nan)


class PallasDirectSum(AbstractForceBackend):
    """`DirectSum` as a fused Pallas kernel, for GPU and TPU.

    Reverse-mode autodiff only. On other backends (e.g. CPU) it falls back to
    `DirectSum`'s code; ``interpret=True`` runs the kernel anywhere, slowly.
    Block sizes default by backend: (16, 64) on GPU, (8, 128) on TPU.

    ``precision="float32"`` evaluates the pair sum in float32 (positions
    recentred on the source centre of mass) and returns the input dtype.
    ``compensated=True`` adds Kahan summation to the forward pass.

    Untested: TPU hardware and AMD GPUs. TPU needs ``precision="float32"``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.nbody as gn
    >>> q = jnp.array([[0.0, 0, 0], [2, 0, 0]])
    >>> s = gn.Sources(q=q, m=jnp.ones(2), h=jnp.zeros(2))
    >>> t = gn.Targets(q=s.q, h=s.h, self_index=jnp.arange(2))
    >>> gn.PallasDirectSum(interpret=True).accelerations(s, t, G=1.0)
    Array([[ 0.25,  0.  ,  0.  ],
           [-0.25,  0.  ,  0.  ]], dtype=float64)

    """

    kernel: AbstractSofteningKernel = PlummerKernel()
    block_targets: int | None = eqx.field(default=None, static=True)
    block_sources: int | None = eqx.field(default=None, static=True)
    num_warps: int | None = eqx.field(default=4, static=True)
    num_stages: int | None = eqx.field(default=None, static=True)
    interpret: bool = eqx.field(default=False, static=True)
    fallback_chunk_size: int = eqx.field(default=1024, static=True)
    precision: str | None = eqx.field(default=None, static=True)
    compensated: bool = eqx.field(default=False, static=True)

    def _use_pallas(self) -> bool:
        return self.interpret or jax.default_backend() in ("gpu", "tpu")

    def _cfg(self) -> _Config:
        bt, bs = (8, 128) if jax.default_backend() == "tpu" else (16, 64)
        return _Config(
            kernel=self.kernel,
            potential=False,
            block_targets=self.block_targets or bt,
            block_sources=self.block_sources or bs,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
            interpret=self.interpret,
            precision=self.precision,
            compensated=self.compensated,
        )

    def prepare(self, sources: Sources) -> Sources:
        """Return ``sources`` unchanged."""
        return sources

    def accelerations(
        self, prepared: Sources, targets: Targets, *, G: float | Array
    ) -> Array:
        """Return the accelerations (Nt, 3) at ``targets``."""
        if self._use_pallas():
            fn = ft.partial(pallas_accelerations, G=G, cfg=self._cfg())
        else:
            fn = ft.partial(
                direct_sum_accelerations,
                kernel=self.kernel,
                G=G,
                chunk_size=self.fallback_chunk_size,
            )
        return self.grad.apply(fn, prepared, targets)

    def potential(
        self, prepared: Sources, targets: Targets, *, G: float | Array
    ) -> Array:
        """Return the potentials (Nt,) at ``targets``."""
        if self._use_pallas():
            fn = ft.partial(pallas_potential, G=G, cfg=self._cfg())
        else:
            fn = ft.partial(
                direct_sum_potential,
                kernel=self.kernel,
                G=G,
                chunk_size=self.fallback_chunk_size,
            )
        return self.grad.apply(fn, prepared, targets)

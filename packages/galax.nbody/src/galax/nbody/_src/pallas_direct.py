"""Direct-sum force backend as a tiled Pallas kernel.

Everything is a pair sum ``out[i] = sum_j fn(A[i], B[j])`` over column
operands ``A`` and row operands ``B``, run by one primitive, `pair_sum_p`.
Each program owns ``block_targets`` rows of ``A`` and loops over ``B`` in tiles
of ``block_sources``, so the pair matrix is never stored.

Autodiff stays inside the primitive, so all modes compose to any order:

- JVP: the tangent is the pair sum of ``jvp(fn)``, linear in the tangents.
- Transpose: the pair sum of ``vjp(fn)``. Cotangents of ``A`` sum over ``B``;
  those of ``B`` sum over ``A``, with the two sides swapped.
- vmap: a leading batch axis on every operand becomes a grid axis.

Padding, NaN targets and ``G`` are handled in plain JAX outside.

Tested on NVIDIA GPUs (Ampere, Ada). TPU lowers but is untested on hardware.
"""

__all__ = ["PallasDirectSum", "pair_sum", "pallas_accelerations", "pallas_potential"]

import dataclasses
import functools as ft

from collections.abc import Callable, Sequence
from jaxtyping import Array
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

from .backend import AbstractForceBackend
from .direct import direct_sum_accelerations, direct_sum_potential
from .kernels import AbstractSofteningKernel, PlummerKernel
from .particles import Sources, Targets

_Arrs = tuple[Array, ...]


def _sds(shape: tuple[int, ...], dtype: Any) -> Any:
    return jax.ShapeDtypeStruct(shape, dtype)  # type: ignore[no-untyped-call]


def _undef(x: Any) -> bool:
    return bool(ad.is_undefined_primal(x))  # type: ignore[no-untyped-call]


def _aval(x: Any) -> Any:
    return x.aval if _undef(x) else jax.typeof(x)


class _Config(NamedTuple):
    """Static configuration."""

    kernel: AbstractSofteningKernel
    potential: bool
    block_targets: int
    block_sources: int
    num_warps: int | None
    num_stages: int | None
    interpret: bool
    precision: str | None = None
    compensated: bool = False


class _Launch(NamedTuple):
    """Launch parameters of `pair_sum_p` (hashable)."""

    block_a: int
    block_b: int
    num_warps: int | None
    num_stages: int | None
    interpret: bool
    compensated: bool = False


# ---------------------------------------------------------------------------
# Pair functions. Called on (BA, 1) columns ``a`` and (1, BB) rows ``b`` with
# global indices ``ia`` (BA, 1) and ``jb`` (1, BB); return a tuple of terms.
# They must be elementwise in the pair, so swapping sides transposes them.
# Frozen dataclasses: hashable by value, so jit caches hit.


@dataclasses.dataclass(frozen=True)
class _Base:
    """Softened gravity. ``a = (x, y, z, h, self_index, ok)``, ``b = (x, y, z, m, h)``.

    Pairs with ``self_index == j`` or ``ok == 0`` are skipped with a double
    ``where``, so every derivative of a skipped pair is finite and zero.
    """

    kernel: AbstractSofteningKernel
    potential: bool

    def __call__(self, a: _Arrs, b: _Arrs, ia: Array, jb: Array) -> _Arrs:
        del ia
        xt, yt, zt, ht, it, okt = a
        xs, ys, zs, ms, hs = b
        dx, dy, dz = xs - xt, ys - yt, zs - zt
        r2 = dx * dx + dy * dy + dz * dz
        skip = (it == jb) | (okt == 0)
        r2s = jnp.where(skip, jnp.ones_like(r2), r2)
        hp = self.kernel.combine(ht, hs)
        if self.potential:
            pot = self.kernel.potential(r2s, hp)
            return (jnp.where(skip, jnp.zeros_like(pot), pot) * ms,)
        f = self.kernel.force_factor(r2s, hp)
        w = jnp.where(skip, jnp.zeros_like(f), f) * ms
        return (w * dx, w * dy, w * dz)


def _fill(
    kept: Sequence[Array],
    zero: tuple[int, ...],
    dtypes: tuple[Any, ...],
    shape: tuple[int, ...],
) -> list[Array]:
    """Reinsert the operands at ``zero`` as zeros of ``shape``."""
    it = iter(kept)
    return [
        jnp.zeros(shape, dt) if k in zero else next(it) for k, dt in enumerate(dtypes)
    ]


@dataclasses.dataclass(frozen=True)
class _Jvp:
    """Tangent of ``f``. Operands: ``f``'s, then tangents of ``nz_a`` / ``nz_b``."""

    f: Any
    n_a: int
    n_b: int
    nz_a: tuple[int, ...]
    nz_b: tuple[int, ...]

    def __call__(self, a: _Arrs, b: _Arrs, ia: Array, jb: Array) -> _Arrs:
        fa, ta = list(a[: self.n_a]), a[self.n_a :]
        fb, tb = list(b[: self.n_b]), b[self.n_b :]

        def g(pa: _Arrs, pb: _Arrs) -> _Arrs:
            xa, xb = list(fa), list(fb)
            for k, v in zip(self.nz_a, pa, strict=True):
                xa[k] = v
            for k, v in zip(self.nz_b, pb, strict=True):
                xb[k] = v
            return tuple(self.f(tuple(xa), tuple(xb), ia, jb))

        prim = (tuple(fa[k] for k in self.nz_a), tuple(fb[k] for k in self.nz_b))
        _, out = jax.jvp(g, prim, (tuple(ta), tuple(tb)))
        return tuple(out)


@dataclasses.dataclass(frozen=True)
class _Vjp:
    """Per-pair cotangents of ``f``'s operands ``wrt`` on ``side``.

    ``side="a"``: operands are ``(f_a kept + ct, f_b kept)``, summed over ``b``.
    ``side="b"``: operands are ``(f_b kept, f_a kept + ct)``, i.e. swapped, so
    the sum runs over ``f``'s ``a``. Operands in ``zero_a`` / ``zero_b`` are
    left out and set to zero (``f`` must be linear in them).
    """

    f: Any
    side: str
    wrt: tuple[int, ...]
    n_ct: int
    dtypes_a: tuple[Any, ...]
    dtypes_b: tuple[Any, ...]
    zero_a: tuple[int, ...] = ()
    zero_b: tuple[int, ...] = ()

    def __call__(self, a: _Arrs, b: _Arrs, ia: Array, jb: Array) -> _Arrs:
        shape = jnp.broadcast_shapes(ia.shape, jb.shape)
        if self.side == "a":
            ka, ct, kb, i, j = a[: -self.n_ct], a[-self.n_ct :], b, ia, jb
        else:
            kb, ka, ct, i, j = a, b[: -self.n_ct], b[-self.n_ct :], jb, ia
        fa = _fill(ka, self.zero_a, self.dtypes_a, shape)
        fb = _fill(kb, self.zero_b, self.dtypes_b, shape)
        mine = fa if self.side == "a" else fb

        def g(*w: Array) -> _Arrs:
            for k, v in zip(self.wrt, w, strict=True):
                mine[k] = v
            out = self.f(tuple(fa), tuple(fb), i, j)
            return tuple(jnp.broadcast_to(o, shape) for o in out)

        x = tuple(jnp.broadcast_to(mine[k], shape) for k in self.wrt)
        out, vjp = jax.vjp(g, *x)
        return tuple(
            vjp(
                tuple(
                    jnp.broadcast_to(c, shape).astype(o.dtype)
                    for c, o in zip(ct, out, strict=True)
                )
            )
        )


@ft.lru_cache(maxsize=512)
def _out_dtypes(fn: Any, da: tuple[Any, ...], db: tuple[Any, ...]) -> tuple[Any, ...]:
    """Output dtypes of ``fn`` on operands of dtypes ``da``, ``db``."""
    sds = lambda dt: _sds((1, 1), dt)
    out = jax.eval_shape(
        lambda a, b: fn(
            a, b, jnp.zeros((1, 1), jnp.int32), jnp.zeros((1, 1), jnp.int32)
        ),
        tuple(map(sds, da)),
        tuple(map(sds, db)),
    )
    return tuple(np.dtype(o.dtype) for o in out)


# ---------------------------------------------------------------------------
# Kernel


def _i32(x: int) -> Array:
    """Loop bounds as int32; Mosaic (TPU) can't lower int64 under x64."""
    return jnp.asarray(x, dtype=jnp.int32)


def _kernel(*refs: Any, fn: Any, n_a: int, n_b: int, nb: int, lc: _Launch) -> None:
    a_refs, b_refs, out_refs = refs[:n_a], refs[n_a : n_a + n_b], refs[n_a + n_b :]
    ba, bb = lc.block_a, lc.block_b
    a = tuple(r[...] for r in a_refs)
    ia = pl.program_id(1) * ba + jax.lax.broadcasted_iota(jnp.int32, (ba, 1), 0)
    shape = (ba, bb)

    def tile(j0: Array | int, *, masked: bool) -> _Arrs:
        b = tuple(r[:, pl.ds(j0, bb)] for r in b_refs)
        jb = j0 + jax.lax.broadcasted_iota(jnp.int32, (1, bb), 1)
        terms = tuple(
            jnp.broadcast_to(x, shape).astype(o.dtype)
            for x, o in zip(fn(a, b, ia, jb), out_refs, strict=True)
        )
        if masked:  # padded rows of B
            terms = tuple(jnp.where(jb < nb, x, jnp.zeros_like(x)) for x in terms)
        return terms

    def add(carry: tuple[_Arrs, _Arrs], terms: _Arrs) -> tuple[_Arrs, _Arrs]:
        acc, comp = carry
        if not lc.compensated:
            return tuple(x + p for x, p in zip(acc, terms, strict=True)), comp
        ys = tuple(p - c for p, c in zip(terms, comp, strict=True))  # Kahan
        new = tuple(x + y for x, y in zip(acc, ys, strict=True))
        comp = tuple((n - x) - y for n, x, y in zip(new, acc, ys, strict=True))
        return new, comp

    # Accumulate per pair; reduce across lanes once at the end.
    zero = tuple(jnp.zeros(shape, o.dtype) for o in out_refs)
    carry = (zero, zero if lc.compensated else ())
    n_full, rem = divmod(nb, bb)
    carry = jax.lax.fori_loop(
        _i32(0),
        _i32(n_full),
        lambda k, c: add(c, tile(k * bb, masked=False)),
        carry,
    )
    if rem:
        carry = add(carry, tile(n_full * bb, masked=True))  # static start
    acc, comp = carry
    if lc.compensated:
        acc = tuple(x - c for x, c in zip(acc, comp, strict=True))
    for o, x in zip(out_refs, acc, strict=True):
        o[...] = jnp.sum(x, axis=1, keepdims=True)


def _pad_to(n: int, b: int) -> int:
    return -(-n // b) * b


def _call(
    *ops: Array, fn: Any, n_a: int, lc: _Launch, compiler_params: Any = None
) -> list[Array]:
    """Run ``fn``'s pair sum. Operands are (L, N) with L = 1 or the batch size."""
    a, b = ops[:n_a], ops[n_a:]
    lead = max(x.shape[0] for x in ops)
    na, nb = a[0].shape[1], b[0].shape[1]
    na_p, nb_p = _pad_to(na, lc.block_a), _pad_to(nb, lc.block_b)
    a = tuple(jnp.pad(x, ((0, 0), (0, na_p - na)))[:, :, None] for x in a)
    b = tuple(jnp.pad(x, ((0, 0), (0, nb_p - nb)))[:, None, :] for x in b)

    def col(x: Array) -> Any:
        one = x.shape[0] == 1
        return pl.BlockSpec((None, lc.block_a, 1), lambda g, i: (0 if one else g, i, 0))

    def row(x: Array) -> Any:
        one = x.shape[0] == 1
        return pl.BlockSpec((None, 1, nb_p), lambda g, _: (0 if one else g, 0, 0))

    dts = _out_dtypes(fn, tuple(x.dtype for x in a), tuple(x.dtype for x in b))
    out_spec = pl.BlockSpec((None, lc.block_a, 1), lambda g, i: (g, i, 0))
    out = pl.pallas_call(
        ft.partial(_kernel, fn=fn, n_a=n_a, n_b=len(b), nb=nb, lc=lc),
        grid=(lead, na_p // lc.block_a),
        in_specs=[col(x) for x in a] + [row(x) for x in b],
        out_specs=[out_spec] * len(dts),
        out_shape=[_sds((lead, na_p, 1), dt) for dt in dts],
        compiler_params=compiler_params,
        interpret=lc.interpret,
        name="nbody_pair_sum",
    )(*a, *b)
    return [x[:, :na, 0] for x in out]


# ---------------------------------------------------------------------------
# Primitive

pair_sum_p = Primitive("nbody_pair_sum")
pair_sum_p.multiple_results = True


def _bind(*ops: Array, **params: Any) -> list[Array]:
    return list(pair_sum_p.bind(*ops, **params))  # type: ignore[no-untyped-call]


def pair_sum(
    fn: Any,
    a: Sequence[Array],
    b: Sequence[Array],
    lc: _Launch,
    *,
    n_lin_a: int = 0,
    n_lin_b: int = 0,
) -> list[Array]:
    """Return ``out[k][g, i] = sum_j fn(a[:][g, i], b[:][g, j])[k]``.

    Operands are (L, N) arrays with L = 1 (shared) or the batch size. The last
    ``n_lin_a`` / ``n_lin_b`` operands of each side are ones ``fn`` is linear in;
    only those can be transposed.
    """
    return _bind(*a, *b, fn=fn, n_a=len(a), n_lin_a=n_lin_a, n_lin_b=n_lin_b, lc=lc)


@ft.lru_cache(maxsize=256)
def _jitted(**params: Any) -> Callable[..., Any]:
    return jax.jit(ft.partial(_bind, **params))


def _impl(*ops: Array, **params: Any) -> list[Array]:
    out: list[Array] = _jitted(**params)(*ops)
    return out


def _abstract_eval(*ops: Any, fn: Any, n_a: int, **_: Any) -> list[Any]:
    a, b = ops[:n_a], ops[n_a:]
    if not a or not b or any(x.ndim != 2 for x in ops):
        msg = "pair_sum needs (L, N) operands on both sides"
        raise ValueError(msg)
    lead = max(x.shape[0] for x in ops)
    dts = _out_dtypes(fn, tuple(x.dtype for x in a), tuple(x.dtype for x in b))
    return [jax.core.ShapedArray((lead, a[0].shape[1]), dt) for dt in dts]


def _gpu_params(lc: _Launch) -> Any:
    from jax.experimental.pallas import triton as pltr

    return pltr.CompilerParams(num_warps=lc.num_warps, num_stages=lc.num_stages)


def _lowering(*, gpu: bool) -> Callable[..., Any]:
    def rule(ctx: Any, *ops: Any, fn: Any, n_a: int, lc: _Launch, **_: Any) -> Any:
        params = _gpu_params(lc) if gpu and not lc.interpret else None
        f = ft.partial(_call, fn=fn, n_a=n_a, lc=lc, compiler_params=params)
        return mlir.lower_fun(f, multiple_results=True)(ctx, *ops)

    return rule


def _jvp(
    primals: Sequence[Array],
    tangents: Sequence[Any],
    *,
    fn: Any,
    n_a: int,
    n_lin_a: int,
    n_lin_b: int,
    lc: _Launch,
) -> tuple[list[Array], list[Any]]:
    out = _bind(*primals, fn=fn, n_a=n_a, n_lin_a=n_lin_a, n_lin_b=n_lin_b, lc=lc)
    nz = [
        type(t) is not ad.Zero and jnp.dtype(t.dtype) != jax.dtypes.float0
        for t in tangents
    ]
    nz_a = tuple(k for k in range(n_a) if nz[k])
    nz_b = tuple(k - n_a for k in range(n_a, len(primals)) if nz[k])
    if not nz_a and not nz_b:
        return out, [ad.Zero.from_primal_value(o) for o in out]
    a, b = primals[:n_a], primals[n_a:]
    ta = [tangents[k] for k in nz_a]
    tb = [tangents[n_a + k] for k in nz_b]
    jfn = _Jvp(fn, n_a, len(b), nz_a, nz_b)
    t_out = pair_sum(
        jfn,
        [*a, *ta],
        [*b, *tb],
        lc._replace(compensated=False),
        n_lin_a=len(ta),
        n_lin_b=len(tb),
    )
    return out, t_out


def _transpose(
    cts: Sequence[Any],
    *ops: Any,
    fn: Any,
    n_a: int,
    n_lin_a: int,
    n_lin_b: int,
    lc: _Launch,
) -> list[Any]:
    n = len(ops)
    lin = set(range(n_a - n_lin_a, n_a)) | set(range(n - n_lin_b, n))
    for k, x in enumerate(ops):
        if _undef(x) and k not in lin:
            msg = "pair_sum can only be transposed in its linear operands"
            raise NotImplementedError(msg)
    res: list[Any] = [None] * n
    if all(type(c) is ad.Zero for c in cts):
        return res
    aval = _aval
    lead = max(aval(x).shape[0] for x in ops)
    ct = [ad.instantiate_zeros(c) for c in cts]  # type: ignore[no-untyped-call]
    ct = [jnp.broadcast_to(c, (lead, c.shape[1])) for c in ct]
    lc = lc._replace(compensated=False)
    und_a = tuple(k for k in range(n_a) if _undef(ops[k]))
    und_b = tuple(k - n_a for k in range(n_a, n) if _undef(ops[k]))

    if isinstance(fn, _Jvp) and (n_lin_a, n_lin_b) == (len(fn.nz_a), len(fn.nz_b)):
        # Transpose of a JVP: VJP of the underlying function at the primal.
        f, nfa = fn.f, fn.n_a
        fa, fb = list(ops[:nfa]), list(ops[n_a : n_a + fn.n_b])
        wrt_a = tuple(fn.nz_a[k - nfa] for k in und_a)
        wrt_b = tuple(fn.nz_b[k - fn.n_b] for k in und_b)
        zero_a: tuple[int, ...] = ()
        zero_b: tuple[int, ...] = ()
    else:
        # Generic linear fn: drop the linear operands (as zeros).
        f, nfa = fn, n_a
        zero_a = tuple(range(n_a - n_lin_a, n_a))
        zero_b = tuple(range(n - n_a - n_lin_b, n - n_a))
        fa = [x for k, x in enumerate(ops[:n_a]) if k not in zero_a]
        fb = [x for k, x in enumerate(ops[n_a:]) if k not in zero_b]
        wrt_a, wrt_b = und_a, und_b

    nfb = n - n_a if f is fn else fn.n_b
    da = tuple(aval(x).dtype for x in ops[:nfa])
    db = tuple(aval(x).dtype for x in ops[n_a : n_a + nfb])
    common: dict[str, Any] = {
        "n_ct": len(ct),
        "dtypes_a": da,
        "dtypes_b": db,
        "zero_a": zero_a,
        "zero_b": zero_b,
    }

    def put(idx: tuple[int, ...], offset: int, vals: Sequence[Array]) -> None:
        for k, v in zip(idx, vals, strict=True):
            target = aval(ops[offset + k])
            shared = target.shape[0] == 1 and v.shape[0] != 1
            res[offset + k] = (v.sum(0, keepdims=True) if shared else v).astype(
                target.dtype
            )

    if wrt_a:
        g = pair_sum(_Vjp(f, "a", wrt_a, **common), [*fa, *ct], fb, lc, n_lin_a=len(ct))
        put(und_a, 0, g)
    if wrt_b:
        g = pair_sum(_Vjp(f, "b", wrt_b, **common), fb, [*fa, *ct], lc, n_lin_b=len(ct))
        put(und_b, n_a, g)
    return res


def _batch(
    args: Sequence[Array],
    dims: Sequence[Any],
    *,
    fn: Any,
    n_a: int,
    n_lin_a: int,
    n_lin_b: int,
    lc: _Launch,
) -> tuple[list[Array], list[int]]:
    size = next(x.shape[d] for x, d in zip(args, dims, strict=True) if d is not None)
    old = max(
        x.shape[0 if d is None or d != 0 else 1]
        for x, d in zip(args, dims, strict=True)
    )
    ops = []
    for x, d in zip(args, dims, strict=True):
        if d is None and x.shape[0] == 1:
            ops.append(x)  # shared by the whole batch
            continue
        y = jnp.moveaxis(x, d, 0) if d is not None else x[None]
        y = jnp.broadcast_to(y, (size, old, y.shape[2]))
        ops.append(y.reshape(size * old, y.shape[2]))
    out = _bind(*ops, fn=fn, n_a=n_a, n_lin_a=n_lin_a, n_lin_b=n_lin_b, lc=lc)
    return [o.reshape(size, old, o.shape[1]) for o in out], [0] * len(out)


pair_sum_p.def_impl(_impl)  # type: ignore[no-untyped-call]
pair_sum_p.def_abstract_eval(_abstract_eval)  # type: ignore[no-untyped-call]
mlir.register_lowering(pair_sum_p, _lowering(gpu=False))
for _plat in ("cuda", "rocm"):
    mlir.register_lowering(pair_sum_p, _lowering(gpu=True), platform=_plat)
ad.primitive_jvps[pair_sum_p] = _jvp
ad.primitive_transposes[pair_sum_p] = _transpose
batching.primitive_batchers[pair_sum_p] = _batch  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Forces


def _run(sources: Sources, targets: Targets, cfg: _Config) -> tuple[Array, Array]:
    """Split into (1, N) operands and run the pair sum."""
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
    qt, qs = qt.astype(dtype), qs.astype(dtype)

    a = (
        *(qt[:, k] for k in range(3)),
        targets.h.astype(dtype),
        targets.self_index.astype(jnp.int32),
        ok.astype(jnp.int32),
    )
    b = (
        *(qs[:, k] for k in range(3)),
        sources.m.astype(dtype),
        sources.h.astype(dtype),
    )
    lc = _Launch(
        block_a=cfg.block_targets,
        block_b=cfg.block_sources,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
        interpret=cfg.interpret,
        compensated=cfg.compensated,
    )
    out = pair_sum(
        _Base(cfg.kernel, cfg.potential), [x[None] for x in a], [x[None] for x in b], lc
    )
    return jnp.stack([o[0] for o in out], axis=1).astype(out_dtype), ok


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

    Supports every autodiff mode (``grad``, ``jvp``, ``linearize``, ``vmap``,
    and higher orders); each derivative is another Pallas pair sum. On other
    backends (e.g. CPU) it falls back to `DirectSum`'s code; ``interpret=True``
    runs the kernel anywhere, slowly. Block sizes default by backend: (16, 64)
    on GPU, (8, 128) on TPU.

    ``precision="float32"`` evaluates the pair sum in float32 (positions
    recentred on the source centre of mass) and returns the input dtype.
    ``compensated=True`` adds Kahan summation to the forward pass. On TPU,
    which has no float64, ``precision`` defaults to ``"float32"``.

    Untested: TPU hardware and AMD GPUs.

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
        tpu = jax.default_backend() == "tpu"
        bt, bs = (8, 128) if tpu else (16, 64)
        return _Config(
            kernel=self.kernel,
            potential=False,
            block_targets=self.block_targets or bt,
            block_sources=self.block_sources or bs,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
            interpret=self.interpret,
            precision=self.precision or ("float32" if tpu else None),
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

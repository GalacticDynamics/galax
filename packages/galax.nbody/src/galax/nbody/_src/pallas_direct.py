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

When a transpose needs cotangents on both sides, `pair_sum_ab_p` can compute
them in one kernel: ``A``'s summed in registers, ``B``'s added atomically
(GPU only; ``_Launch.fused``). Elsewhere it lowers to the two pair sums.

Padding, NaN targets and ``G`` are handled in plain JAX outside.

Tested on NVIDIA GPUs (Ampere, Ada). TPU lowers but is untested on hardware.
"""

__all__ = ["PallasDirectSum", "pair_sum", "pallas_accelerations", "pallas_potential"]

import contextlib
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


def _sds(shape: tuple[int, ...], dtype: Any, vma: frozenset[str] = frozenset()) -> Any:
    return jax.ShapeDtypeStruct(shape, dtype, vma=vma)  # type: ignore[no-untyped-call]


def _no_check_vma() -> Any:
    from jax._src import config  # private: no public switch

    state = getattr(config, "_check_vma", None)
    return contextlib.nullcontext() if state is None else state(False)  # noqa: FBT003


def _vma(avals: Sequence[Any]) -> frozenset[str]:
    """Mesh axes an output varies over under `shard_map`: any operand's."""
    return frozenset().union(*(getattr(x, "vma", frozenset()) for x in avals))


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
    fused_grad: bool | None = None


class _Launch(NamedTuple):
    """Launch parameters of `pair_sum_p` (hashable)."""

    block_a: int
    block_b: int
    num_warps: int | None
    num_stages: int | None
    interpret: bool
    compensated: bool = False
    fused: bool = False
    split: bool = False  # set at lowering: skip masking off the self tiles


# ---------------------------------------------------------------------------
# Pair functions. Called on (BA, 1) columns ``a`` and (1, BB) rows ``b`` with
# global indices ``ia`` (BA, 1) and ``jb`` (1, BB); return a tuple of terms.
# They must be elementwise in the pair, so swapping sides transposes them.
# ``exact_tiles(a, bb, n)`` gives the range of ``b`` tiles that may hold a
# skipped pair for this block of ``a`` (None: all); elsewhere the kernel calls
# them with ``exact=False``, which skips the masking.
# Frozen dataclasses: hashable by value, so jit caches hit.


@dataclasses.dataclass(frozen=True)
class _Base:
    """Softened gravity. ``a = (x, y, z, h, self_index, ok)``, ``b = (x, y, z, m, h)``.

    Pairs with ``self_index == j`` or ``ok == 0`` are skipped with a double
    ``where``, so every derivative of a skipped pair is finite and zero.
    ``exact=False`` drops that masking, for tiles without such pairs.
    """

    kernel: AbstractSofteningKernel
    potential: bool

    def __call__(
        self, a: _Arrs, b: _Arrs, ia: Array, jb: Array, *, exact: bool = True
    ) -> _Arrs:
        del ia
        xt, yt, zt, ht, it, okt = a
        xs, ys, zs, ms, hs = b
        dx, dy, dz = xs - xt, ys - yt, zs - zt
        r2 = dx * dx + dy * dy + dz * dz
        skip = (it == jb) | (okt == 0)
        keep: Callable[[Array], Array] = lambda x: (
            jnp.where(skip, jnp.zeros_like(x), x) if exact else x
        )
        r2s = jnp.where(skip, jnp.ones_like(r2), r2) if exact else r2
        hp = self.kernel.combine(ht, hs)
        if self.potential:
            return (keep(self.kernel.potential(r2s, hp)) * ms,)
        w = keep(self.kernel.force_factor(r2s, hp)) * ms
        return (w * dx, w * dy, w * dz)

    def exact_tiles(self, a: _Arrs, bb: int, n: int) -> tuple[Array, Array] | None:
        """``[lo, hi)``: tiles with this block's self pairs; all if any ``ok == 0``."""
        it, okt = a[4], a[5]
        has = it >= 0
        lo = jnp.min(jnp.where(has, it, jnp.full_like(it, n * bb))) // bb
        hi = jnp.max(jnp.where(has, it, jnp.full_like(it, -bb))) // bb + 1
        bad = jnp.max((okt == 0).astype(jnp.int32)) > 0  # no reduce_or in Triton
        lo = jnp.where(bad, jnp.zeros_like(lo), jnp.clip(lo, 0, n))
        hi = jnp.where(bad, jnp.full_like(hi, n), jnp.clip(hi, lo, n))
        return lo, hi


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

    def exact_tiles(self, a: _Arrs, bb: int, n: int) -> tuple[Array, Array] | None:
        return _exact_tiles(self.f, a[: self.n_a], bb, n)

    def __call__(
        self, a: _Arrs, b: _Arrs, ia: Array, jb: Array, *, exact: bool = True
    ) -> _Arrs:
        fa, ta = list(a[: self.n_a]), a[self.n_a :]
        fb, tb = list(b[: self.n_b]), b[self.n_b :]

        def g(pa: _Arrs, pb: _Arrs) -> _Arrs:
            xa, xb = list(fa), list(fb)
            for k, v in zip(self.nz_a, pa, strict=True):
                xa[k] = v
            for k, v in zip(self.nz_b, pb, strict=True):
                xb[k] = v
            return tuple(self.f(tuple(xa), tuple(xb), ia, jb, exact=exact))

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

    def exact_tiles(self, a: _Arrs, bb: int, n: int) -> tuple[Array, Array] | None:
        if self.side == "b":  # f's a operands are columns here
            return None
        fa = _fill(a[: -self.n_ct], self.zero_a, self.dtypes_a, (1, 1))
        return _exact_tiles(self.f, tuple(fa), bb, n)

    def __call__(
        self, a: _Arrs, b: _Arrs, ia: Array, jb: Array, *, exact: bool = True
    ) -> _Arrs:
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
            out = self.f(tuple(fa), tuple(fb), i, j, exact=exact)
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


@dataclasses.dataclass(frozen=True)
class _Both:
    """Per-pair cotangents of ``f``'s ``wrt_a`` and ``wrt_b`` operands, one VJP.

    Operands as for ``_Vjp(side="a")``: ``(f_a kept + ct, f_b kept)``. Returns
    the ``wrt_a`` cotangents, then the ``wrt_b`` ones, all per pair.
    """

    f: Any
    wrt_a: tuple[int, ...]
    wrt_b: tuple[int, ...]
    n_ct: int
    dtypes_a: tuple[Any, ...]
    dtypes_b: tuple[Any, ...]
    zero_a: tuple[int, ...] = ()
    zero_b: tuple[int, ...] = ()

    def exact_tiles(self, a: _Arrs, bb: int, n: int) -> tuple[Array, Array] | None:
        fa = _fill(a[: -self.n_ct], self.zero_a, self.dtypes_a, (1, 1))
        return _exact_tiles(self.f, tuple(fa), bb, n)

    def __call__(
        self, a: _Arrs, b: _Arrs, ia: Array, jb: Array, *, exact: bool = True
    ) -> _Arrs:
        shape = jnp.broadcast_shapes(ia.shape, jb.shape)
        ka, ct = a[: -self.n_ct], a[-self.n_ct :]
        fa = _fill(ka, self.zero_a, self.dtypes_a, shape)
        fb = _fill(b, self.zero_b, self.dtypes_b, shape)

        def g(xa: _Arrs, xb: _Arrs) -> _Arrs:
            for k, v in zip(self.wrt_a, xa, strict=True):
                fa[k] = v
            for k, v in zip(self.wrt_b, xb, strict=True):
                fb[k] = v
            out = self.f(tuple(fa), tuple(fb), ia, jb, exact=exact)
            return tuple(jnp.broadcast_to(o, shape) for o in out)

        xa = tuple(jnp.broadcast_to(fa[k], shape) for k in self.wrt_a)
        xb = tuple(jnp.broadcast_to(fb[k], shape) for k in self.wrt_b)
        out, vjp = jax.vjp(g, xa, xb)
        ga, gb = vjp(
            tuple(
                jnp.broadcast_to(c, shape).astype(o.dtype)
                for c, o in zip(ct, out, strict=True)
            )
        )
        return (*ga, *gb)


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


def _exact_tiles(fn: Any, a: _Arrs, bb: int, n: int) -> tuple[Array, Array] | None:
    get = getattr(fn, "exact_tiles", None)
    return None if get is None else get(a, bb, n)


def _loop(
    fn: Any,
    a: _Arrs,
    n: int,
    bb: int,
    step: Callable[..., Any],
    carry: Any,
    *,
    split: bool,
) -> Any:
    """Run ``step(k, carry, exact=...)`` over tiles ``k < n``, exact only where needed.

    Three plain loops rather than a branch per tile: a ``cond`` in the loop
    body slows the Triton kernel down more than the masking costs. Off on TPU:
    Mosaic fails to lower the data-dependent loop bounds.
    """
    rng = _exact_tiles(fn, a, bb, n) if split else None
    if rng is None:
        return jax.lax.fori_loop(
            _i32(0), _i32(n), lambda k, c: step(k, c, exact=True), carry
        )
    lo, hi = (x.astype(jnp.int32) for x in rng)
    carry = jax.lax.fori_loop(_i32(0), lo, lambda k, c: step(k, c, exact=False), carry)
    carry = jax.lax.fori_loop(lo, hi, lambda k, c: step(k, c, exact=True), carry)
    return jax.lax.fori_loop(hi, _i32(n), lambda k, c: step(k, c, exact=False), carry)


def _i32(x: int) -> Array:
    """Loop bounds as int32; Mosaic (TPU) can't lower int64 under x64."""
    return jnp.asarray(x, dtype=jnp.int32)


def _kernel(*refs: Any, fn: Any, n_a: int, n_b: int, nb: int, lc: _Launch) -> None:
    a_refs, b_refs, out_refs = refs[:n_a], refs[n_a : n_a + n_b], refs[n_a + n_b :]
    ba, bb = lc.block_a, lc.block_b
    a = tuple(r[...] for r in a_refs)
    ia = pl.program_id(1) * ba + jax.lax.broadcasted_iota(jnp.int32, (ba, 1), 0)
    shape = (ba, bb)

    def tile(j0: Array | int, *, masked: bool, exact: bool = True) -> _Arrs:
        b = tuple(r[:, pl.ds(j0, bb)] for r in b_refs)
        jb = j0 + jax.lax.broadcasted_iota(jnp.int32, (1, bb), 1)
        out = fn(a, b, ia, jb) if exact else fn(a, b, ia, jb, exact=False)
        terms = tuple(
            jnp.broadcast_to(x, shape).astype(o.dtype)
            for x, o in zip(out, out_refs, strict=True)
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
    carry = _loop(
        fn,
        a,
        n_full,
        bb,
        lambda k, c, exact: add(c, tile(k * bb, masked=False, exact=exact)),
        carry,
        split=lc.split,
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
    vma = _vma([jax.typeof(x) for x in ops])
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
        out_shape=[_sds((lead, na_p, 1), dt, vma) for dt in dts],
        compiler_params=compiler_params,
        interpret=lc.interpret,
        name="nbody_pair_sum",
    )(*a, *b)
    return [x[:, :na, 0] for x in out]


def _kernel_ab(
    *refs: Any, fn: _Both, n_a: int, n_b: int, na: int, nb: int, lc: _Launch
) -> None:
    """One VJP per pair: ``A``'s cotangents summed here, ``B``'s added atomically."""
    from jax.experimental.pallas import triton as pltr

    n_ga, n_gb = len(fn.wrt_a), len(fn.wrt_b)
    a_refs, b_refs = refs[:n_a], refs[n_a : n_a + n_b]
    out_a = refs[n_a + n_b + n_gb : n_a + n_b + n_gb + n_ga]  # after aliased zeros
    out_b = refs[n_a + n_b + n_gb + n_ga :]
    ba, bb = lc.block_a, lc.block_b
    a = tuple(r[...] for r in a_refs)
    ia = pl.program_id(1) * ba + jax.lax.broadcasted_iota(jnp.int32, (ba, 1), 0)
    shape = (ba, bb)

    def tile(j0: Array | int, acc: _Arrs, *, masked: bool, exact: bool = True) -> _Arrs:
        b = tuple(r[:, pl.ds(j0, bb)] for r in b_refs)
        jb = j0 + jax.lax.broadcasted_iota(jnp.int32, (1, bb), 1)
        g = fn(a, b, ia, jb) if exact else fn(a, b, ia, jb, exact=False)
        ok = ia < na
        if masked:
            ok = ok & (jb < nb)
        ga, gb = (
            tuple(
                jnp.where(ok, jnp.broadcast_to(x, shape), jnp.zeros(shape, o.dtype))
                for x, o in zip(xs, outs, strict=True)
            )
            for xs, outs in ((g[:n_ga], out_a), (g[n_ga:], out_b))
        )
        idx = (pl.ds(_i32(0), 1), pl.ds(jnp.asarray(j0, jnp.int32), bb))  # int32 (x64)
        for o, x in zip(out_b, gb, strict=True):
            pltr.atomic_add(o, idx, x.sum(0, keepdims=True))
        return tuple(s + x for s, x in zip(acc, ga, strict=True))

    acc = tuple(jnp.zeros(shape, o.dtype) for o in out_a)
    n_full, rem = divmod(nb, bb)
    acc = _loop(
        fn,
        a,
        n_full,
        bb,
        lambda k, c, exact: tile(k * bb, c, masked=False, exact=exact),
        acc,
        split=lc.split,
    )
    if rem:
        acc = tile(n_full * bb, acc, masked=True)
    for o, x in zip(out_a, acc, strict=True):
        o[...] = jnp.sum(x, axis=1, keepdims=True)


def _call_ab(
    *ops: Array, fn: _Both, n_a: int, lc: _Launch, compiler_params: Any = None
) -> list[Array]:
    """Run `_kernel_ab`. Operands as for `_call`."""
    a, b = ops[:n_a], ops[n_a:]
    lead = max(x.shape[0] for x in ops)
    vma = _vma([jax.typeof(x) for x in ops])
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

    n_ga = len(fn.wrt_a)
    dts = _out_dtypes(fn, tuple(x.dtype for x in a), tuple(x.dtype for x in b))
    zeros = [jnp.zeros((lead, 1, nb_p), dt) for dt in dts[n_ga:]]
    if vma:  # aliased to outputs, so they must vary like them
        zeros = [jax.lax.pvary(z, tuple(sorted(vma))) for z in zeros]  # type: ignore[no-untyped-call]
    out_a = pl.BlockSpec((None, lc.block_a, 1), lambda g, i: (g, i, 0))
    n_in = len(a) + len(b)
    out = pl.pallas_call(
        ft.partial(_kernel_ab, fn=fn, n_a=n_a, n_b=len(b), na=na, nb=nb, lc=lc),
        grid=(lead, na_p // lc.block_a),
        in_specs=[col(x) for x in a] + [row(x) for x in (*b, *zeros)],
        out_specs=[out_a] * n_ga + [row(z) for z in zeros],
        out_shape=[_sds((lead, na_p, 1), dt, vma) for dt in dts[:n_ga]]
        + [_sds(z.shape, z.dtype, vma) for z in zeros],
        input_output_aliases={n_in + k: n_ga + k for k in range(len(zeros))},
        compiler_params=compiler_params,
        interpret=lc.interpret,
        name="nbody_pair_sum_fused",
    )(*a, *b, *zeros)
    return [x[:, :na, 0] for x in out[:n_ga]] + [x[:, 0, :nb] for x in out[n_ga:]]


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
    return _out_avals(ops, [(lead, a[0].shape[1])] * len(dts), dts)


def _out_avals(
    ops: Sequence[Any], shapes: Sequence[tuple[int, ...]], dts: Sequence[Any]
) -> list[Any]:
    """Output avals, varying over every mesh axis an operand varies over."""
    vma = _vma(ops)
    sharding = None
    if vma:  # the operands' (manual) mesh
        mesh = next(x.sharding.mesh for x in ops if x.vma)
        spec = jax.sharding.PartitionSpec()  # type: ignore[no-untyped-call]
        sharding = jax.sharding.NamedSharding(mesh, spec)
    return [
        jax.core.ShapedArray(s, dt, sharding=sharding, vma=vma)
        for s, dt in zip(shapes, dts, strict=True)
    ]


def _gpu_params(lc: _Launch) -> Any:
    from jax.experimental.pallas import triton as pltr

    return pltr.CompilerParams(num_warps=lc.num_warps, num_stages=lc.num_stages)


def _lowering(*, gpu: bool) -> Callable[..., Any]:
    def rule(ctx: Any, *ops: Any, fn: Any, n_a: int, lc: _Launch, **_: Any) -> Any:
        params = _gpu_params(lc) if gpu and not lc.interpret else None
        lc = lc._replace(split=gpu or lc.interpret)
        f = ft.partial(_call, fn=fn, n_a=n_a, lc=lc, compiler_params=params)
        # The Pallas interpreter loses `vma` on refs carried through loops, so
        # `shard_map`'s check fails inside it; lowering doesn't need it.
        with _no_check_vma() if lc.interpret else contextlib.nullcontext():
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
    t_out = _tangent(primals, tangents, fn=fn, n_a=n_a, lc=lc)
    if t_out is None:
        return out, [ad.Zero.from_primal_value(o) for o in out]
    return out, t_out


def _tangent(
    primals: Sequence[Array], tangents: Sequence[Any], *, fn: Any, n_a: int, lc: _Launch
) -> list[Array] | None:
    """Tangent of ``pair_sum(fn, ...)``, or None if all ``tangents`` are zero."""
    nz = [
        type(t) is not ad.Zero and jnp.dtype(t.dtype) != jax.dtypes.float0
        for t in tangents
    ]
    nz_a = tuple(k for k in range(n_a) if nz[k])
    nz_b = tuple(k - n_a for k in range(n_a, len(primals)) if nz[k])
    if not nz_a and not nz_b:
        return None
    a, b = primals[:n_a], primals[n_a:]
    ta = [tangents[k] for k in nz_a]
    tb = [tangents[n_a + k] for k in nz_b]
    jfn = _Jvp(fn, n_a, len(b), nz_a, nz_b)
    return pair_sum(
        jfn,
        [*a, *ta],
        [*b, *tb],
        lc._replace(compensated=False),
        n_lin_a=len(ta),
        n_lin_b=len(tb),
    )


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

    if wrt_a and wrt_b and lc.fused:
        g = _bind_ab(
            *fa,
            *ct,
            *fb,
            fn=_Both(f, wrt_a, wrt_b, **common),
            fn_a=_Vjp(f, "a", wrt_a, **common),
            fn_b=_Vjp(f, "b", wrt_b, **common),
            n_fa=len(fa),
            n_ct=len(ct),
            lc=lc,
        )
        put(und_a, 0, g[: len(wrt_a)])
        put(und_b, n_a, g[len(wrt_a) :])
        return res
    if wrt_a:
        g = pair_sum(_Vjp(f, "a", wrt_a, **common), [*fa, *ct], fb, lc, n_lin_a=len(ct))
        put(und_a, 0, g)
    if wrt_b:
        g = pair_sum(_Vjp(f, "b", wrt_b, **common), fb, [*fa, *ct], lc, n_lin_b=len(ct))
        put(und_b, n_a, g)
    return res


def _batch(
    args: Sequence[Array], dims: Sequence[Any], **params: Any
) -> tuple[list[Array], list[int]]:
    return _batch_with(_bind, args, dims, params)


def _batch_with(
    bind: Callable[..., list[Array]],
    args: Sequence[Array],
    dims: Sequence[Any],
    params: dict[str, Any],
) -> tuple[list[Array], list[int]]:
    """Fold a batch axis into the leading (grid) axis of every operand."""
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
    out = bind(*ops, **params)
    return [o.reshape(size, old, o.shape[1]) for o in out], [0] * len(out)


pair_sum_p.def_impl(_impl)  # type: ignore[no-untyped-call]
pair_sum_p.def_abstract_eval(_abstract_eval)  # type: ignore[no-untyped-call]
mlir.register_lowering(pair_sum_p, _lowering(gpu=False))
for _plat in ("cuda", "rocm"):
    mlir.register_lowering(pair_sum_p, _lowering(gpu=True), platform=_plat)
ad.primitive_jvps[pair_sum_p] = _jvp
ad.primitive_transposes[pair_sum_p] = _transpose
batching.primitive_batchers[pair_sum_p] = _batch  # type: ignore[assignment]


# ``(*f_a, *ct, *f_b) -> (cotangents of wrt_a, cotangents of wrt_b)``: the
# pair sums of ``fn_a`` and ``fn_b`` (two `_Vjp`s), fused on GPU via ``fn``.
pair_sum_ab_p = Primitive("nbody_pair_sum_ab")
pair_sum_ab_p.multiple_results = True


def _bind_ab(*ops: Array, **params: Any) -> list[Array]:
    return list(pair_sum_ab_p.bind(*ops, **params))  # type: ignore[no-untyped-call]


def _ab_parts(
    ops: Sequence[Any], *, n_fa: int, n_ct: int
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """``ops`` reordered as the operands of the two unfused pair sums."""
    fa, ct, fb = ops[:n_fa], ops[n_fa : n_fa + n_ct], ops[n_fa + n_ct :]
    return (*fa, *ct, *fb), (*fb, *fa, *ct)


def _ab_two_pass(
    *ops: Array, fn: _Both, fn_a: _Vjp, fn_b: _Vjp, n_fa: int, n_ct: int, lc: _Launch
) -> list[Array]:
    del fn
    pa, pb = _ab_parts(ops, n_fa=n_fa, n_ct=n_ct)
    nb = len(ops) - n_fa - n_ct
    ga = _bind(*pa, fn=fn_a, n_a=n_fa + n_ct, n_lin_a=n_ct, n_lin_b=0, lc=lc)
    gb = _bind(*pb, fn=fn_b, n_a=nb, n_lin_a=0, n_lin_b=n_ct, lc=lc)
    return [*ga, *gb]


def _ab_abstract_eval(*ops: Any, fn: _Both, n_fa: int, n_ct: int, **_: Any) -> Any:
    n_a = n_fa + n_ct
    a, b = ops[:n_a], ops[n_a:]
    lead = max(x.shape[0] for x in ops)
    dts = _out_dtypes(fn, tuple(x.dtype for x in a), tuple(x.dtype for x in b))
    n_ga = len(fn.wrt_a)
    shapes = [(lead, (a if k < n_ga else b)[0].shape[1]) for k in range(len(dts))]
    return _out_avals(ops, shapes, dts)


@ft.lru_cache(maxsize=256)
def _jitted_ab(**params: Any) -> Callable[..., Any]:
    return jax.jit(ft.partial(_bind_ab, **params))


def _ab_impl(*ops: Array, **params: Any) -> list[Array]:
    out: list[Array] = _jitted_ab(**params)(*ops)
    return out


def _ab_lowering(*, gpu: bool) -> Callable[..., Any]:
    def rule(ctx: Any, *ops: Any, **params: Any) -> Any:
        lc = params["lc"]
        if gpu and lc.fused and not lc.interpret:
            f: Any = ft.partial(
                _call_ab,
                fn=params["fn"],
                n_a=params["n_fa"] + params["n_ct"],
                lc=lc._replace(split=True),
                compiler_params=_gpu_params(lc),
            )
        else:
            f = ft.partial(_ab_two_pass, **params)
        return mlir.lower_fun(f, multiple_results=True)(ctx, *ops)

    return rule


def _ab_jvp(
    primals: Sequence[Array], tangents: Sequence[Any], **params: Any
) -> tuple[list[Array], list[Any]]:
    """Tangents via the two unfused pair sums (their transposes fuse again)."""
    out = _bind_ab(*primals, **params)
    n_fa, n_ct, lc = params["n_fa"], params["n_ct"], params["lc"]
    n_ga = len(params["fn"].wrt_a)
    parts = zip(
        _ab_parts(primals, n_fa=n_fa, n_ct=n_ct),
        _ab_parts(tangents, n_fa=n_fa, n_ct=n_ct),
        (params["fn_a"], params["fn_b"]),
        (n_fa + n_ct, len(primals) - n_fa - n_ct),
        (out[:n_ga], out[n_ga:]),
        strict=True,
    )
    t_out: list[Any] = []
    for p, t, fn, n_a, o in parts:
        t_part = _tangent(p, t, fn=fn, n_a=n_a, lc=lc)
        t_out += [ad.Zero.from_primal_value(x) for x in o] if t_part is None else t_part
    return out, t_out


def _ab_transpose(cts: Sequence[Any], *ops: Any, **params: Any) -> list[Any]:
    """Linear in ``ct`` only; transpose the two-pass form."""
    und = [k for k, x in enumerate(ops) if _undef(x)]

    def f(*xs: Array) -> list[Array]:
        full = list(ops)
        for k, x in zip(und, xs, strict=True):
            full[k] = x
        return _ab_two_pass(*full, **params)

    avals = [
        _sds(ops[k].aval.shape, ops[k].aval.dtype, _vma([ops[k].aval])) for k in und
    ]
    g = jax.linear_transpose(f, *avals)(
        [ad.instantiate_zeros(c) for c in cts]  # type: ignore[no-untyped-call]
    )
    res: list[Any] = [None] * len(ops)
    for k, v in zip(und, g, strict=True):
        res[k] = v
    return res


pair_sum_ab_p.def_impl(_ab_impl)  # type: ignore[no-untyped-call]
pair_sum_ab_p.def_abstract_eval(_ab_abstract_eval)  # type: ignore[no-untyped-call]
mlir.register_lowering(pair_sum_ab_p, _ab_lowering(gpu=False))
for _plat in ("cuda", "rocm"):
    mlir.register_lowering(pair_sum_ab_p, _ab_lowering(gpu=True), platform=_plat)
ad.primitive_jvps[pair_sum_ab_p] = _ab_jvp
ad.primitive_transposes[pair_sum_ab_p] = _ab_transpose
batching.primitive_batchers[pair_sum_ab_p] = lambda args, dims, **p: _batch_with(  # type: ignore[assignment]
    _bind_ab, args, dims, p
)


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
        fused=jnp.dtype(dtype) == jnp.float64
        if cfg.fused_grad is None
        else cfg.fused_grad,
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

    ``fused_grad``: on GPU, compute reverse-mode cotangents of targets and
    sources in one pass, adding the sources' atomically (~1.8x faster in
    float64; their summation order, so the last bits, can vary between runs).
    ``None`` (default) turns it on when the pair sum runs in float64.

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
    fused_grad: bool | None = eqx.field(default=None, static=True)

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
            fused_grad=self.fused_grad,
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

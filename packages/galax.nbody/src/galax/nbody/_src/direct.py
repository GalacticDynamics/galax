"""Direct-sum force backend, chunked over targets and checkpointed."""

__all__ = ["DirectSum", "direct_sum_accelerations", "direct_sum_potential"]

import functools as ft
import operator

from collections.abc import Callable
from jaxtyping import Array

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import AbstractForceBackend, Sources, Targets
from .kernels import AbstractSofteningKernel, PlummerKernel


def _pair_terms(
    sources: Sources,
    x_i: Array,
    h_i: Array,
    self_index_i: Array,
    combine: Callable[[Array, Array], Array],
    pair_fn: Callable[[Array, Array], Array],
) -> tuple[Array, Array, Array]:
    """Return ``d = x_j - x_i``, the pair terms, and whether ``x_i`` is finite.

    The pair term is ``pair_fn(r2, combine(h_i, h_j))``. The self pair gets
    ``r2 = 1`` before ``pair_fn`` so its value and gradient stay finite at zero
    softening, then the outer ``where`` zeroes it.

    A non-finite ``x_i`` is replaced by 0 and the caller sets that row's result
    to NaN. Masking inside the row keeps the NaN out of the gradients of every
    other row: a zero cotangent times a NaN is still NaN.

    ponytail: a target at zero distance from a source with zero pair softening
    still gives NaN gradients for the sources. Masking those pairs costs about
    60% of the forward pass at N=4096; add it if exact coincidences matter.
    """
    ok = jnp.all(jnp.isfinite(x_i))
    d = sources.q - jnp.where(ok, x_i, 0.0)
    r2 = jnp.sum(d * d, -1)
    self_pair = jnp.arange(r2.shape[0]) == self_index_i
    r2s = jnp.where(self_pair, 1.0, r2)
    w = jnp.where(self_pair, 0.0, pair_fn(r2s, combine(h_i, sources.h)))
    return d, w, ok


def direct_sum_accelerations(
    sources: Sources,
    targets: Targets,
    *,
    kernel: AbstractSofteningKernel,
    G: float | Array,
    chunk_size: int,
) -> Array:
    """Return the accelerations (Nt, 3) of ``targets`` from ``sources``.

    Targets are mapped in chunks of ``chunk_size`` and each row is
    checkpointed, so reverse-mode memory scales as ``chunk_size * Ns``. A
    target with a non-finite position gets NaN, and that NaN stays out of the
    gradients of the other targets.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.nbody as gn
    >>> from galax.nbody._src.direct import direct_sum_accelerations
    >>> q = jnp.array([[0.0, 0, 0], [2, 0, 0]])
    >>> s = gn.Sources(q=q, m=jnp.ones(2), h=jnp.zeros(2))
    >>> t = gn.Targets(q=s.q, h=s.h, self_index=jnp.arange(2))
    >>> direct_sum_accelerations(s, t, kernel=gn.PlummerKernel(), G=1.0, chunk_size=1)
    Array([[ 0.25,  0.  ,  0.  ],
           [-0.25,  0.  ,  0.  ]], dtype=float64)

    """

    def row_fn(args: tuple[Array, Array, Array]) -> Array:
        x_i, h_i, self_index_i = args
        d, w, ok = _pair_terms(
            sources, x_i, h_i, self_index_i, kernel.combine, kernel.force_factor
        )
        return jnp.where(ok, G * jnp.sum((sources.m * w)[:, None] * d, 0), jnp.nan)

    out: Array = jax.lax.map(
        jax.checkpoint(row_fn),
        (targets.q, targets.h, targets.self_index),
        batch_size=chunk_size,
    )
    return out


def direct_sum_potential(
    sources: Sources,
    targets: Targets,
    *,
    kernel: AbstractSofteningKernel,
    G: float | Array,
    chunk_size: int,
) -> Array:
    """Return the potentials (Nt,) of ``targets`` from ``sources``.

    Self pairs contribute 0. Bad targets get NaN as in
    `direct_sum_accelerations`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.nbody as gn
    >>> from galax.nbody._src.direct import direct_sum_potential
    >>> q = jnp.array([[0.0, 0, 0], [2, 0, 0]])
    >>> s = gn.Sources(q=q, m=jnp.ones(2), h=jnp.zeros(2))
    >>> t = gn.Targets(q=s.q, h=s.h, self_index=jnp.arange(2))
    >>> direct_sum_potential(s, t, kernel=gn.PlummerKernel(), G=1.0, chunk_size=1)
    Array([-0.5, -0.5], dtype=float64)

    """

    def row_fn(args: tuple[Array, Array, Array]) -> Array:
        x_i, h_i, self_index_i = args
        _, w, ok = _pair_terms(
            sources, x_i, h_i, self_index_i, kernel.combine, kernel.potential
        )
        return jnp.where(ok, G * jnp.sum(sources.m * w), jnp.nan)

    out: Array = jax.lax.map(
        jax.checkpoint(row_fn),
        (targets.q, targets.h, targets.self_index),
        batch_size=chunk_size,
    )
    return out


def _as_chunk_size(c: object) -> int:
    """Return ``c`` as a positive `int`; reject bools, floats and values <= 0."""
    msg = f"chunk_size must be a positive int, got {c!r}"
    if isinstance(c, bool):
        raise ValueError(msg)  # noqa: TRY004
    try:
        n = operator.index(c)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError(msg) from None
    if n <= 0:
        raise ValueError(msg)
    return n


class DirectSum(AbstractForceBackend):
    """Exact O(Nt * Ns) pairwise sum, chunked over targets.

    ``kernel`` and ``G`` are closed over by the function handed to the gradient
    policy. Under ``Autodiff`` a traced ``G`` differentiates correctly. Under
    ``NoGrad``, differentiating with respect to a traced ``G`` or kernel fails.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.nbody as gn
    >>> q = jnp.array([[0.0, 0, 0], [2, 0, 0]])
    >>> s = gn.Sources(q=q, m=jnp.ones(2), h=jnp.zeros(2))
    >>> t = gn.Targets(q=s.q, h=s.h, self_index=jnp.arange(2))
    >>> b = gn.DirectSum(grad="none")
    >>> b.potential(b.prepare(s), t, G=1.0)
    Array([-0.5, -0.5], dtype=float64)

    """

    kernel: AbstractSofteningKernel = PlummerKernel()
    chunk_size: int = eqx.field(default=1024, static=True, converter=_as_chunk_size)

    def prepare(self, sources: Sources) -> Sources:
        """Return ``sources`` unchanged."""
        return sources

    def accelerations(
        self, prepared: Sources, targets: Targets, *, G: float | Array
    ) -> Array:
        """Return the accelerations (Nt, 3) at ``targets``."""
        fn = ft.partial(
            direct_sum_accelerations,
            kernel=self.kernel,
            G=G,
            chunk_size=self.chunk_size,
        )
        return self.grad.apply(fn, prepared, targets)

    def potential(
        self, prepared: Sources, targets: Targets, *, G: float | Array
    ) -> Array:
        """Return the potentials (Nt,) at ``targets``."""
        fn = ft.partial(
            direct_sum_potential, kernel=self.kernel, G=G, chunk_size=self.chunk_size
        )
        return self.grad.apply(fn, prepared, targets)

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
    pair_fn: Callable[[Array, Array], Array],
) -> tuple[Array, Array]:
    """Return ``d = x_j - x_i`` and ``pair_fn(r2, h_ij)`` with self pairs zeroed.

    The self pair gets ``r2 = 1`` before ``pair_fn`` so its value and gradient
    stay finite at zero softening, then the outer ``where`` zeroes it.
    """
    d = sources.q - x_i
    r2 = jnp.sum(d * d, -1)
    self_pair = jnp.arange(r2.shape[0]) == self_index_i
    r2s = jnp.where(self_pair, 1.0, r2)
    return d, jnp.where(self_pair, 0.0, pair_fn(r2s, h_i))


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
    checkpointed, so reverse-mode memory scales as ``chunk_size * Ns``.

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
        d, w = _pair_terms(
            sources,
            x_i,
            h_i,
            self_index_i,
            lambda r2, h: kernel.force_factor(r2, kernel.combine(h, sources.h)),
        )
        return G * jnp.sum((sources.m * w)[:, None] * d, axis=0)

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

    Self pairs contribute 0.

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
        _, w = _pair_terms(
            sources,
            x_i,
            h_i,
            self_index_i,
            lambda r2, h: kernel.potential(r2, kernel.combine(h, sources.h)),
        )
        return G * jnp.sum(sources.m * w)

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

"""galax: Galactic Dynamix in Jax."""


__all__ = [
    "partial_jit",
    "partial_vmap",
    "partial_vectorize",
    "vectorize_method",
]

from collections.abc import Callable, Hashable, Iterable, Sequence
from functools import partial
from typing import Any, NotRequired, TypedDict, TypeVar

import jax
from typing_extensions import ParamSpec, Unpack

P = ParamSpec("P")
R = TypeVar("R")


class JITKwargs(TypedDict):
    """Keyword arguments for :func:`jax.jit`."""

    in_shardings: NotRequired[Any]
    out_shardings: NotRequired[Any]
    static_argnums: NotRequired[int | Sequence[int] | None]
    static_argnames: NotRequired[str | Iterable[str] | None]
    donate_argnums: NotRequired[int | Sequence[int] | None]
    donate_argnames: NotRequired[str | Iterable[str] | None]
    keep_unused: NotRequired[bool]
    device: NotRequired[jax.Device | None]
    backend: NotRequired[str | None]
    inline: NotRequired[bool]


def partial_jit(
    **kwargs: Unpack[JITKwargs],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    return partial(jax.jit, **kwargs)


# TODO: nest the definitions properly
class VMapKwargs(TypedDict):
    in_axes: NotRequired[int | Sequence[Any] | dict[str, Any] | None]
    out_axes: NotRequired[Any]
    axis_name: NotRequired[Hashable | None]
    axis_size: NotRequired[int | None]
    spmd_axis_name: NotRequired[Hashable | tuple[Hashable, ...] | None]


def partial_vmap(
    **kwargs: Unpack[VMapKwargs],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    return partial(jax.vmap, **kwargs)


class VectorizeKwargs(TypedDict):
    excluded: NotRequired[Sequence[int] | None]
    signature: NotRequired[str | None]


def partial_vectorize(
    **kwargs: Unpack[VectorizeKwargs],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    return partial(jax.numpy.vectorize, **kwargs)


def vectorize_method(
    **kwargs: Unpack[VectorizeKwargs],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a method to :func:`jax.numpy.vectorize`.

    This is a wrapper around :func:`jax.numpy.vectorize` that vectorizes a
    class' method by returning a :class:`functools.partial`. It is equivalent to
    :func:`partial_vectorize`, except that ``excluded`` is set to exclude the
    0th argument (``self``). As a result, the ``excluded`` tuple should start
    at 0 to exclude the first 'real' argument (proceeding ``self``).
    """
    # Prepend 0 to excluded to exclude the first argument (self)
    excluded = tuple(kwargs.get("excluded") or (-1,))  # (None -> (0,))
    excluded = excluded if excluded[0] == -1 else (-1, *excluded)
    kwargs["excluded"] = tuple(i + 1 for i in excluded)

    return partial_vectorize(**kwargs)

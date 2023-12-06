"""galdynamix: Galactic Dynamix in Jax."""


__all__ = ["partial_jit", "partial_vmap", "partial_vectorize"]

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
    **kwargs: Unpack[VMapKwargs],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    return partial(jax.numpy.vectorize, **kwargs)

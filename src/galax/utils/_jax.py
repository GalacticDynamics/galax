"""galax: Galactic Dynamix in Jax."""


__all__ = [
    "partial_jit",
    "partial_vmap",
    "partial_vectorize",
    "vectorize_method",
]

from ast import TypeAlias
from collections.abc import Callable, Hashable, Iterable, Sequence
from functools import partial
from typing import Any, NotRequired, TypedDict, TypeVar

import jax
from typing_extensions import ParamSpec, Unpack

P = ParamSpec("P")
R = TypeVar("R")
CallPR: TypeAlias = Callable[P, R]


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


def partial_jit(**kw: Unpack[JITKwargs]) -> Callable[[CallPR], CallPR]:
    """Decorate a function with :func:`jax.jit`.

    Parameters
    ----------
    **kw : Unpack[JITKwargs]
        Keyword arguments for :func:`jax.jit`.
        See :func:`jax.jit` for more information.

    Returns
    -------
    :class:`~functools.partial`
        A partial function to :func:`jax.jit` a function.
    """
    return partial(jax.jit, **kw)


# TODO: nest the definitions properly
class VMapKwargs(TypedDict):
    """Keyword arguments for :func:`jax.vmap`."""

    in_axes: NotRequired[int | Sequence[Any] | dict[str, Any] | None]
    out_axes: NotRequired[Any]
    axis_name: NotRequired[Hashable | None]
    axis_size: NotRequired[int | None]
    spmd_axis_name: NotRequired[Hashable | tuple[Hashable, ...] | None]


def partial_vmap(**kw: Unpack[VMapKwargs]) -> Callable[[CallPR], CallPR]:
    """Decorate a function with :func:`jax.vmap`.

    Parameters
    ----------
    **kw : Unpack[VMapKwargs]
        Keyword arguments for :func:`jax.vmap`.
        See :func:`jax.vmap` for more information.

    Returns
    -------
    :class:`~functools.partial`
        A partial function to :func:`jax.vmap` a function.
    """
    return partial(jax.vmap, **kw)


class VectorizeKwargs(TypedDict):
    """Keyword arguments for :func:`jax.numpy.vectorize`."""

    excluded: NotRequired[Sequence[int] | None]
    signature: NotRequired[str | None]


def partial_vectorize(**kw: Unpack[VectorizeKwargs]) -> Callable[[CallPR], CallPR]:
    """Decorate a function with :func:`jax.numpy.vectorize`.

    Parameters
    ----------
    **kw : Unpack[VMapKwargs]
        Keyword arguments for :func:`jax.numpy.vectorize`.
        See :func:`jax.numpy.vectorize` for more information.

    Returns
    -------
    :class:`~functools.partial`
        A partial function to :func:`jax.numpy.vectorize` a function.
    """
    return partial(jax.numpy.vectorize, **kw)


def vectorize_method(**kw: Unpack[VectorizeKwargs]) -> Callable[[CallPR], CallPR]:
    """:func:`jax.numpy.vectorize` a class' method.

    This is a wrapper around :func:`jax.numpy.vectorize` that vectorizes a
    class' method by returning a :class:`functools.partial`. It is equivalent to
    :func:`partial_vectorize`, except that ``excluded`` is set to exclude the
    0th argument (``self``). As a result, the ``excluded`` tuple should start
    at 0 to exclude the first 'real' argument (proceeding ``self``).

    Parameters
    ----------
    **kw : Unpack[VMapKwargs]
        Keyword arguments for :func:`jax.numpy.vectorize`.
        See :func:`jax.numpy.vectorize` for more information.
    """
    # Prepend 0 to excluded to exclude the first argument (self)
    excluded = tuple(kw.get("excluded") or (-1,))  # (None -> (0,))
    excluded = excluded if excluded[0] == -1 else (-1, *excluded)
    kw["excluded"] = tuple(i + 1 for i in excluded)

    return partial_vectorize(**kw)

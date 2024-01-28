"""galax: Galactic Dynamix in Jax."""


__all__ = ["vectorize_method"]

from collections.abc import Callable, Sequence
from functools import partial
from typing import NotRequired, TypedDict, TypeVar

import jax
from typing_extensions import ParamSpec, Unpack

P = ParamSpec("P")
R = TypeVar("R")


class VectorizeKwargs(TypedDict):
    """Keyword arguments for :func:`jax.numpy.vectorize`."""

    excluded: NotRequired[Sequence[int] | None]
    signature: NotRequired[str | None]


def vectorize_method(
    **kw: Unpack[VectorizeKwargs],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """:func:`jax.numpy.vectorize` a class' method.

    This is a wrapper around :func:`jax.numpy.vectorize` that vectorizes a
    class' method by returning a :class:`functools.partial`.  ``excluded`` is
    set to exclude the 0th argument (``self``). As a result, the ``excluded``
    tuple should start at 0 to exclude the first 'real' argument (proceeding
    ``self``).

    Parameters
    ----------
    **kw : Unpack[VMapKwargs]
        Keyword arguments for :func:`jax.numpy.vectorize`.  See
        :func:`jax.numpy.vectorize` for more information.
    """
    # Prepend 0 to excluded to exclude the first argument (self)
    excluded = tuple(kw.get("excluded") or (-1,))  # (None -> (0,))
    excluded = excluded if excluded[0] == -1 else (-1, *excluded)
    kw["excluded"] = tuple(i + 1 for i in excluded)

    return partial(jax.numpy.vectorize, **kw)

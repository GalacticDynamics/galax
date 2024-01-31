"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from typing import Any, Protocol, cast, runtime_checkable

import jax.experimental.array_api as xp
from jaxtyping import Array, Float


@runtime_checkable
class Shaped(Protocol):
    """Protocol for a shaped object."""

    shape: tuple[int, ...]


def _getitem_time_index_tuple(index: tuple[Any, ...], t: Float[Array, "..."]) -> Any:
    """Get the time index from a slice."""
    if len(index) == 0:  # slice is an empty tuple
        return slice(None)
    if t.ndim == 1:  # slicing a Vec1
        return slice(None)
    if len(index) >= t.ndim:
        msg = f"Index {index} has too many dimensions for time array of shape {t.shape}"
        raise IndexError(msg)
    return index


def _getitem_time_index_shaped(index: Shaped, t: Float[Array, "..."]) -> Shaped:
    """Get the time index from a slice."""
    if t.ndim == 1:  # Vec1
        return cast(Shaped, xp.asarray([True]))
    if len(index.shape) >= t.ndim:
        msg = f"Index {index} has too many dimensions for time array of shape {t.shape}"
        raise IndexError(msg)
    return index


def getitem_time_index(index: Any, t: Float[Array, "..."]) -> Any:
    """Get the time index from an index."""
    if isinstance(index, tuple):
        return _getitem_time_index_tuple(index, t)
    if isinstance(index, Shaped):
        return _getitem_time_index_shaped(index, t)
    return index

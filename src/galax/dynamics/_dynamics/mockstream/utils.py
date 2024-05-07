"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from typing import Any, Protocol, TypeVar, cast, runtime_checkable

import jax
from jaxtyping import Array, Bool

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class SupportsGetItem(Protocol[T_co]):
    """Protocol for types that support the `__getitem__` method."""

    def __getitem__(self, key: Any) -> T_co: ...


def _identity(x: T) -> T:
    return x


def _reverse(x: SupportsGetItem[T]) -> T:
    return x[::-1]


def cond_reverse(pred: Bool[Array, ""], x: T) -> T:
    return cast(T, jax.lax.cond(pred, _reverse, _identity, x))

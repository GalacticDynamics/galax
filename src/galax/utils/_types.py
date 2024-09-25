"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from typing import Any, Protocol, TypeVar, runtime_checkable

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class SupportsGetItem(Protocol[T_co]):
    """Protocol for types that support the `__getitem__` method."""

    def __getitem__(self, key: Any) -> T_co: ...

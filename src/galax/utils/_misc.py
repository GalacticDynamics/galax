"""galax: Galactic Dynamix in Jax."""


__all__: list[str] = []

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def first(x: Iterable[T], /) -> T:
    """Return first element from iterable."""
    return next(iter(x))

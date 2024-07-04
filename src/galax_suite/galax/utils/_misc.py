"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


# TODO: make a mini-package called `zeroth` and move this function there
def zeroth(x: Iterable[T], /) -> T:
    """Return first element from iterable."""
    return next(iter(x))

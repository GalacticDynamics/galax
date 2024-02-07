"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from functools import partial
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

import jax
import jax.experimental.array_api as xp
from jaxtyping import Array, Bool, Shaped

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@partial(jax.jit, static_argnames="axis")
def interleave_concat(
    a: Shaped[Array, "..."], b: Shaped[Array, "..."], /, axis: int
) -> Shaped[Array, "..."]:
    a_shp = a.shape
    return xp.stack((a, b), axis=axis + 1).reshape(
        *a_shp[:axis], 2 * a_shp[axis], *a_shp[axis + 1 :]
    )


# -------------------------------------------------------------------


@runtime_checkable
class SupportsGetItem(Protocol[T_co]):
    """Protocol for types that support the `__getitem__` method."""

    def __getitem__(self, key: Any) -> T_co:
        ...


def _identity(x: T) -> T:
    return x


def _reverse(x: SupportsGetItem[T]) -> T:
    return x[::-1]


def cond_reverse(pred: Bool[Array, ""], x: T) -> T:
    return cast(T, jax.lax.cond(pred, _reverse, _identity, x))

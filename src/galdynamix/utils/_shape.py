"""galdynamix: Galactic Dynamix in Jax."""

__all__: list[str] = []

from typing import overload

import jax.numpy as xp
from jaxtyping import Array, ArrayLike

from galdynamix.typing import ArrayAnyShape

from ._jax import partial_jit


@overload
def atleast_batched() -> tuple[Array, ...]:
    ...


@overload
def atleast_batched(x: ArrayLike, /) -> Array:
    ...


@overload
def atleast_batched(
    x: ArrayLike, y: ArrayLike, /, *arys: ArrayLike
) -> tuple[Array, ...]:
    ...


@partial_jit()
def atleast_batched(*arys: ArrayLike) -> Array | tuple[Array, ...]:
    if len(arys) == 1:
        arr = xp.asarray(arys[0])
        if arr.ndim >= 2:
            return arr
        if arr.ndim == 1:
            return xp.expand_dims(arr, axis=1)
        return xp.expand_dims(arr, axis=(0, 1))
    return tuple(atleast_batched(arr) for arr in arys)


def batched_shape(
    arr: ArrayAnyShape, /, *, expect_scalar: bool
) -> tuple[tuple[int, ...], int]:
    """Return the shape of the batch dimensions of an array."""
    if arr.ndim == 0:
        raise NotImplementedError
    if arr.ndim == 1:
        return (arr.shape, 1) if expect_scalar else ((), arr.shape[0])
    return arr.shape[:-1], arr.shape[-1]

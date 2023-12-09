"""galdynamix: Galactic Dynamix in Jax."""

__all__: list[str] = []

from typing import Literal, overload

import jax.numpy as xp
from jaxtyping import Array, ArrayLike

from galdynamix.typing import ArrayAnyShape, FloatLike

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


@overload
def batched_shape(
    arr: ArrayAnyShape | FloatLike, /, *, expect_ndim: Literal[0]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ...


@overload
def batched_shape(
    arr: ArrayAnyShape | FloatLike, /, *, expect_ndim: Literal[1]
) -> tuple[tuple[int, ...], tuple[int]]:
    ...


@overload
def batched_shape(
    arr: ArrayAnyShape | FloatLike, /, *, expect_ndim: Literal[2]
) -> tuple[tuple[int, ...], tuple[int, int]]:
    ...


@overload
def batched_shape(
    arr: ArrayAnyShape | FloatLike, /, *, expect_ndim: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ...


def batched_shape(
    arr: ArrayAnyShape | FloatLike, /, *, expect_ndim: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the (batch_shape, arr_shape) an array.

    Parameters
    ----------
    arr : array-like
        The array to get the shape of.
    expect_ndim : int
        The expected dimensionality of the array.

    Returns
    -------
    batch_shape : tuple[int, ...]
        The shape of the batch.
    arr_shape : tuple[int, ...]
        The shape of the array.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from galdynamix.utils._shape import batched_shape

    Expecting a scalar:
    >>> batched_shape(0, expect_ndim=0)
    ((), ())
    >>> batched_shape(jnp.array([1]), expect_ndim=0)
    ((1,), ())
    >>> batched_shape(jnp.array([1, 2, 3]), expect_ndim=0)
    ((3,), ())

    Expecting a 1D vector:
    >>> batched_shape(jnp.array(0), expect_ndim=1)
    ((), (1,))
    >>> batched_shape(jnp.array([1]), expect_ndim=1)
    ((), (1,))
    >>> batched_shape(jnp.array([1, 2, 3]), expect_ndim=1)
    ((), (3,))
    >>> batched_shape(jnp.array([[1, 2, 3]]), expect_ndim=1)
    ((1,), (3,))

    Expecting a 2D matrix:
    >>> batched_shape(jnp.array([[1]]), expect_ndim=2)
    ((), (1, 1))
    >>> batched_shape(jnp.array([[[1]]]), expect_ndim=2)
    ((1,), (1, 1))
    >>> batched_shape(jnp.array([[[1]], [[1]]]), expect_ndim=2)
    ((2,), (1, 1))
    """
    shape: tuple[int, ...] = xp.shape(arr)
    ndim = len(shape)
    return shape[: ndim - expect_ndim], shape[ndim - expect_ndim :]

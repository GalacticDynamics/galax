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


# =============================================================================


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


def expand_batch_dims(arr: ArrayAnyShape, /, ndim: int) -> ArrayAnyShape:
    """Expand the batch dimensions of an array.

    Parameters
    ----------
    arr : array-like
        The array to expand the batch dimensions of.
    ndim : int
        The number of batch dimensions to expand.

    Returns
    -------
    arr : array-like
        The array with expanded batch dimensions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from galdynamix.utils._shape import expand_batch_dims

    >>> expand_batch_dims(jnp.array(0), ndim=0).shape
    ()

    >>> expand_batch_dims(jnp.array([0]), ndim=0).shape
    (1,)

    >>> expand_batch_dims(jnp.array(0), ndim=1).shape
    (1,)

    >>> expand_batch_dims(jnp.array([0, 1]), ndim=1).shape
    (1, 2)
    """
    return xp.expand_dims(arr, axis=tuple(0 for _ in range(ndim)))


def expand_arr_dims(arr: ArrayAnyShape, /, ndim: int) -> ArrayAnyShape:
    """Expand the array dimensions of an array.

    Parameters
    ----------
    arr : array-like
        The array to expand the array dimensions of.
    ndim : int
        The number of array dimensions to expand.

    Returns
    -------
    arr : array-like
        The array with expanded array dimensions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from galdynamix.utils._shape import expand_arr_dims

    >>> expand_arr_dims(jnp.array(0), ndim=0).shape
    ()

    >>> expand_arr_dims(jnp.array([0]), ndim=0).shape
    (1,)

    >>> expand_arr_dims(jnp.array(0), ndim=1).shape
    (1,)

    >>> expand_arr_dims(jnp.array([0, 0]), ndim=1).shape
    (2, 1)
    """
    nbatch = len(arr.shape)
    return xp.expand_dims(arr, axis=tuple(nbatch + i for i in range(ndim)))

"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from functools import partial
from typing import Any, Literal, NoReturn, overload

import array_api_jax_compat as xp
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from vector import AbstractVectorBase

from galax.typing import AnyScalar, ArrayAnyShape
from galax.utils._jax import quaxify


@overload
def atleast_batched() -> NoReturn:
    ...


@overload
def atleast_batched(x: ArrayLike, /) -> Array:
    ...


@overload
def atleast_batched(
    x: ArrayLike, y: ArrayLike, /, *arys: ArrayLike
) -> tuple[Array, ...]:
    ...


@quaxify
@partial(jax.jit)
def atleast_batched(*arys: Any) -> Array | tuple[Array, ...]:
    """Convert inputs to arrays with at least two dimensions.

    Parameters
    ----------
    *arys : array_like
        One or more array-like sequences. Non-array inputs are converted to
        arrays. Arrays that already have two or more dimensions are preserved.

    Returns
    -------
    res : tuple
        A tuple of arrays, each with ``a.ndim >= 2``. Copies are made only if
        necessary.

    Examples
    --------
    >>> from galax.utils._shape import atleast_batched
    >>> atleast_batched(0)
    Array([[0]], dtype=int64, ...)

    >>> atleast_batched([1])
    Array([[1]], dtype=int64)

    >>> atleast_batched([[1]])
    Array([[1]], dtype=int64)

    >>> atleast_batched([[[1]]])
    Array([[[1]]], dtype=int64)

    >>> atleast_batched([1, 2, 3])
    Array([[1],
           [2],
           [3]], dtype=int64)

    >>> import jax.numpy as jnp
    >>> jnp.atleast_2d(xp.asarray([1, 2, 3]))
    Array([[1, 2, 3]], dtype=int64)
    """
    if len(arys) == 0:
        msg = "atleast_batched() requires at least one argument"
        raise ValueError(msg)
    if len(arys) == 1:
        arr = jnp.asarray(arys[0])
        if arr.ndim >= 2:
            return arr
        if arr.ndim == 1:
            return xp.expand_dims(arr, axis=1)
        return jnp.expand_dims(arr, axis=(0, 1))
    return tuple(atleast_batched(arr) for arr in arys)


# =============================================================================


def vector_batched_shape(obj: AbstractVectorBase) -> tuple[tuple[int, ...], tuple[int]]:
    """Return the batch and component shape of a vector."""
    return obj.shape, (len(obj.components),)


@overload
def batched_shape(
    arr: ArrayAnyShape | AnyScalar, /, *, expect_ndim: Literal[0]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ...


@overload
def batched_shape(
    arr: ArrayAnyShape | AnyScalar, /, *, expect_ndim: Literal[1]
) -> tuple[tuple[int, ...], tuple[int]]:
    ...


@overload
def batched_shape(
    arr: ArrayAnyShape | AnyScalar, /, *, expect_ndim: Literal[2]
) -> tuple[tuple[int, ...], tuple[int, int]]:
    ...


@overload
def batched_shape(
    arr: ArrayAnyShape | AnyScalar, /, *, expect_ndim: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ...


@quaxify
def batched_shape(
    arr: ArrayAnyShape | AnyScalar | float | int, /, *, expect_ndim: int
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
    Standard imports:

        >>> import array_api_jax_compat as xp
        >>> from galax.utils._shape import batched_shape

    Expecting a scalar:

        >>> batched_shape(0, expect_ndim=0)
        ((), ())
        >>> batched_shape(xp.asarray([1]), expect_ndim=0)
        ((1,), ())
        >>> batched_shape(xp.asarray([1, 2, 3]), expect_ndim=0)
        ((3,), ())

    Expecting a 1D vector:

        >>> batched_shape(xp.asarray(0), expect_ndim=1)
        ((), ())
        >>> batched_shape(xp.asarray([1]), expect_ndim=1)
        ((), (1,))
        >>> batched_shape(xp.asarray([1, 2, 3]), expect_ndim=1)
        ((), (3,))
        >>> batched_shape(xp.asarray([[1, 2, 3]]), expect_ndim=1)
        ((1,), (3,))

    Expecting a 2D matrix:

        >>> batched_shape(xp.asarray([[1]]), expect_ndim=2)
        ((), (1, 1))
        >>> batched_shape(xp.asarray([[[1]]]), expect_ndim=2)
        ((1,), (1, 1))
        >>> batched_shape(xp.asarray([[[1]], [[1]]]), expect_ndim=2)
        ((2,), (1, 1))
    """
    shape: tuple[int, ...] = xp.asarray(arr).shape
    ndim = len(shape)
    return shape[: ndim - expect_ndim], shape[ndim - expect_ndim :]


@quaxify
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
    >>> from galax.utils._shape import expand_batch_dims

    >>> expand_batch_dims(jnp.array(0), ndim=0).shape
    ()

    >>> expand_batch_dims(jnp.array([0]), ndim=0).shape
    (1,)

    >>> expand_batch_dims(jnp.array(0), ndim=1).shape
    (1,)

    >>> expand_batch_dims(jnp.array([0, 1]), ndim=1).shape
    (1, 2)
    """
    return jnp.expand_dims(arr, axis=tuple(range(ndim)))


@quaxify
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
    >>> from galax.utils._shape import expand_arr_dims

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
    return jnp.expand_dims(arr, axis=tuple(nbatch + i for i in range(ndim)))

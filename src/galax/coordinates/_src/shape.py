"""Shape utilities. Private API.

`coordinates` is the lowest portion that uses these, so they live here and
`potential` / `dynamics` import them from here.
"""

__all__: tuple[str, ...] = ()

from jaxtyping import Array, Shaped
from typing import Literal, TypeAlias, overload

import quax
from quax import quaxify

import coordinax as cx
import quaxed.numpy as jnp

import galax._custom_types as gt

AnyScalar: TypeAlias = Shaped[Array, ""]
ArrayAnyShape: TypeAlias = Shaped[Array | quax.ArrayValue, "..."]


def vector_batched_shape(obj: cx.vecs.AbstractVector, /) -> tuple[gt.Shape, int]:
    """Return the batch and component shape of a vector."""
    return obj.shape, len(obj.components)


@overload
def batched_shape(
    arr: ArrayAnyShape | AnyScalar, /, *, expect_ndim: Literal[0]
) -> tuple[gt.Shape, gt.Shape]: ...


@overload
def batched_shape(
    arr: ArrayAnyShape | AnyScalar, /, *, expect_ndim: Literal[1]
) -> tuple[gt.Shape, tuple[int]]: ...


@overload
def batched_shape(
    arr: ArrayAnyShape | AnyScalar, /, *, expect_ndim: Literal[2]
) -> tuple[gt.Shape, tuple[int, int]]: ...


@overload
def batched_shape(
    arr: ArrayAnyShape | AnyScalar, /, *, expect_ndim: int
) -> tuple[gt.Shape, gt.Shape]: ...


@quaxify
def batched_shape(
    arr: ArrayAnyShape | AnyScalar | float | int, /, *, expect_ndim: int
) -> tuple[gt.Shape, gt.Shape]:
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

        >>> import quaxed.numpy as jnp
        >>> from galax.coordinates._src.shape import batched_shape

    Expecting a scalar:

        >>> batched_shape(0, expect_ndim=0)
        ((), ())
        >>> batched_shape(jnp.asarray([1]), expect_ndim=0)
        ((1,), ())
        >>> batched_shape(jnp.asarray([1, 2, 3]), expect_ndim=0)
        ((3,), ())

    Expecting a 1D vector:

        >>> batched_shape(jnp.asarray(0), expect_ndim=1)
        ((), ())
        >>> batched_shape(jnp.asarray([1]), expect_ndim=1)
        ((), (1,))
        >>> batched_shape(jnp.asarray([1, 2, 3]), expect_ndim=1)
        ((), (3,))
        >>> batched_shape(jnp.asarray([[1, 2, 3]]), expect_ndim=1)
        ((1,), (3,))

    Expecting a 2D matrix:

        >>> batched_shape(jnp.asarray([[1]]), expect_ndim=2)
        ((), (1, 1))
        >>> batched_shape(jnp.asarray([[[1]]]), expect_ndim=2)
        ((1,), (1, 1))
        >>> batched_shape(jnp.asarray([[[1]], [[1]]]), expect_ndim=2)
        ((2,), (1, 1))
    """
    shape: gt.Shape = jnp.asarray(arr).shape
    ndim = len(shape)
    return shape[: ndim - expect_ndim], shape[ndim - expect_ndim :]

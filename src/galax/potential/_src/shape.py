"""Shape utilities. Private API.

`potential` is the only portion that expands array dimensions, so these live
here rather than in `coordinates` alongside `batched_shape`.
"""

__all__: tuple[str, ...] = ()

import quaxed.numpy as jnp

from galax.coordinates._src.shape import ArrayAnyShape, quaxify


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
    >>> from galax.potential._src.shape import expand_batch_dims

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
    >>> from galax.potential._src.shape import expand_arr_dims

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

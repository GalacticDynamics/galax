"""Test the `galax.potential._src.shape` module."""

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

from galax.potential._src.shape import expand_arr_dims, expand_batch_dims


class TestExpandBatchDims:
    """Test :func:`galax.potential._src.shape.expand_batch_dims`."""

    @pytest.mark.parametrize(
        ("arr", "ndim", "expect"),
        [
            # ArrayLike
            (jnp.asarray(1), 0, jnp.asarray(1)),
            (jnp.asarray([2]), 0, jnp.asarray([2])),
            (jnp.asarray([1, 2]), 0, jnp.asarray([1, 2])),
            (jnp.asarray(1), 1, jnp.asarray([1])),
            (jnp.asarray([2]), 1, jnp.asarray([[2]])),
            (jnp.asarray([1, 2]), 1, jnp.asarray([[1, 2]])),
            (jnp.asarray(1), 2, jnp.asarray([[1]])),
            # Quantity
            (u.Q(1, "m"), 0, u.Q(1, "m")),
            (u.Q([2], "m"), 0, u.Q([2], "m")),
            (u.Q([1, 2], "m"), 0, u.Q([1, 2], "m")),
            (u.Q(1, "m"), 1, u.Q([1], "m")),
            (u.Q([2], "m"), 1, u.Q([[2]], "m")),
            (u.Q([1, 2], "m"), 1, u.Q([[1, 2]], "m")),
            (u.Q(1, "m"), 2, u.Q([[1]], "m")),
        ],
    )
    def test_expand_batch_dims(
        self, arr: jax.Array, ndim: int, expect: jax.Array
    ) -> None:
        """Test :func:`galax.potential._src.shape.expand_batch_dims`."""
        got = expand_batch_dims(arr, ndim=ndim)
        assert jnp.array_equal(got, expect)
        assert got.shape == expect.shape


class TestExpandArrDims:
    """Test :func:`galax.potential._src.shape.expand_arr_dims`."""

    @pytest.mark.parametrize(
        ("arr", "ndim", "expect"),
        [
            # ArrayLike
            (jnp.asarray(1), 0, jnp.asarray(1)),
            (jnp.asarray([2]), 0, jnp.asarray([2])),
            (jnp.asarray([1, 2]), 0, jnp.asarray([1, 2])),
            (jnp.asarray(1), 1, jnp.asarray([1])),
            (jnp.asarray([2]), 1, jnp.asarray([[2]])),
            (jnp.asarray([1, 2]), 1, jnp.asarray([[1], [2]])),
            (jnp.asarray(1), 2, jnp.asarray([[1]])),
            # Quantity
            (u.Q(1, "m"), 0, u.Q(1, "m")),
            (u.Q([2], "m"), 0, u.Q([2], "m")),
            (u.Q([1, 2], "m"), 0, u.Q([1, 2], "m")),
            (u.Q(1, "m"), 1, u.Q([1], "m")),
            (u.Q([2], "m"), 1, u.Q([[2]], "m")),
            (u.Q([1, 2], "m"), 1, u.Q([[1], [2]], "m")),
            (u.Q(1, "m"), 2, u.Q([[1]], "m")),
        ],
    )
    def test_expand_arr_dims(
        self, arr: jax.Array, ndim: int, expect: jax.Array
    ) -> None:
        """Test :func:`galax.potential._src.shape.expand_arr_dims`."""
        got = expand_arr_dims(arr, ndim=ndim)
        assert jnp.array_equal(got, expect)
        assert got.shape == expect.shape

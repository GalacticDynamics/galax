"""Test the `galax.utils._shape` module."""

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
from galax.utils._shape import batched_shape, expand_arr_dims, expand_batch_dims


class TestBatchedShape:
    """Test the `galax.utils._shape.batched_shape` function."""

    @pytest.mark.parametrize(
        ("arr", "expect_ndim", "expect"),
        [
            # ArrayLike
            (jnp.asarray(42), 0, ((), ())),
            (jnp.asarray([1]), 0, ((1,), ())),
            (jnp.asarray([1, 2, 3]), 1, ((), (3,))),
            (jnp.asarray([[1, 2], [3, 4]]), 1, ((2,), (2,))),
            (jnp.asarray([[1, 2], [3, 4]]), 2, ((), (2, 2))),
            # Quantity
            (u.Quantity(42, "m"), 0, ((), ())),
            (u.Quantity([1], "m"), 0, ((1,), ())),
            (u.Quantity([1, 2, 3], "m"), 1, ((), (3,))),
            (u.Quantity([[1, 2], [3, 4]], "m"), 1, ((2,), (2,))),
            (u.Quantity([[1, 2], [3, 4]], "m"), 2, ((), (2, 2))),
        ],
    )
    def test_batched_shape(
        self, arr: jax.Array, expect_ndim: int, expect: tuple[gt.Shape, gt.Shape]
    ) -> None:
        """Test the `galax.utils._shape.batched_shape` function."""
        batch, shape = batched_shape(arr, expect_ndim=expect_ndim)
        assert batch == expect[0]
        assert shape == expect[1]


class TestExpandBatchDims:
    """Test :func:`galax.utils._shape.expand_batch_dims`."""

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
            (u.Quantity(1, "m"), 0, u.Quantity(1, "m")),
            (u.Quantity([2], "m"), 0, u.Quantity([2], "m")),
            (u.Quantity([1, 2], "m"), 0, u.Quantity([1, 2], "m")),
            (u.Quantity(1, "m"), 1, u.Quantity([1], "m")),
            (u.Quantity([2], "m"), 1, u.Quantity([[2]], "m")),
            (u.Quantity([1, 2], "m"), 1, u.Quantity([[1, 2]], "m")),
            (u.Quantity(1, "m"), 2, u.Quantity([[1]], "m")),
        ],
    )
    def test_expand_batch_dims(
        self, arr: jax.Array, ndim: int, expect: jax.Array
    ) -> None:
        """Test :func:`galax.utils._shape.expand_batch_dims`."""
        got = expand_batch_dims(arr, ndim=ndim)
        assert jnp.array_equal(got, expect)
        assert got.shape == expect.shape


class TestExpandArrDims:
    """Test :func:`galax.utils._shape.expand_arr_dims`."""

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
            (u.Quantity(1, "m"), 0, u.Quantity(1, "m")),
            (u.Quantity([2], "m"), 0, u.Quantity([2], "m")),
            (u.Quantity([1, 2], "m"), 0, u.Quantity([1, 2], "m")),
            (u.Quantity(1, "m"), 1, u.Quantity([1], "m")),
            (u.Quantity([2], "m"), 1, u.Quantity([[2]], "m")),
            (u.Quantity([1, 2], "m"), 1, u.Quantity([[1], [2]], "m")),
            (u.Quantity(1, "m"), 2, u.Quantity([[1]], "m")),
        ],
    )
    def test_expand_arr_dims(
        self, arr: jax.Array, ndim: int, expect: jax.Array
    ) -> None:
        """Test :func:`galax.utils._shape.expand_arr_dims`."""
        got = expand_arr_dims(arr, ndim=ndim)
        assert jnp.array_equal(got, expect)
        assert got.shape == expect.shape

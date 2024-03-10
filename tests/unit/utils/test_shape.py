"""Test the `galax.utils._shape` module."""

import re
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax.numpy import array_equal
from quax import quaxify

import quaxed.array_api as xp
from jax_quantity import Quantity

from galax.utils._shape import (
    atleast_batched,
    batched_shape,
    expand_arr_dims,
    expand_batch_dims,
)

array_equal = quaxify(array_equal)


class TestAtleastBatched:
    """Test the `atleast_batched` function."""

    def test_atleast_batched_no_args(self) -> None:
        """Test the `atleast_batched` function with no arguments."""
        with pytest.raises(
            ValueError,
            match=re.escape("atleast_batched() requires at least one argument"),
        ):
            _ = atleast_batched()

    def test_atleast_batched_example(self) -> None:
        """Test the `atleast_batched` function with an example."""
        x = xp.asarray([1, 2, 3])
        # `atleast_batched` versus `atleast_2d`
        assert array_equal(atleast_batched(x), x[:, None])
        assert array_equal(jnp.atleast_2d(x), x[None, :])

    @pytest.mark.parametrize(
        ("x", "expect"),
        [
            # ArrayLike
            (0, [[0]]),  # scalar -> 2D array
            ([1], [[1]]),  # 1D array -> 2D array
            ([1, 2, 3], [[1], [2], [3]]),
            ([[1]], [[1]]),  # 2D array -> 2D array
            ([[[1]]], [[[1]]]),  # 3D array -> 3D array
            # Quantity
            (Quantity(1, "m"), Quantity([[1]], "m")),  # scalar -> 2D array
            (Quantity([1], "m"), Quantity([[1]], "m")),  # 1D array -> 2D array
            (Quantity([1, 2, 3], "m"), Quantity([[1], [2], [3]], "m")),
            (Quantity([[1]], "m"), Quantity([[1]], "m")),  # 2D array -> 2D array
            (Quantity([[[1]]], "m"), Quantity([[[1]]], "m")),  # 3D array -> 3D array
        ],
    )
    def test_atleast_batched_one_arg(self, x: Any, expect: Any) -> None:
        """Test the `atleast_batched` function with one argument."""
        got = atleast_batched(xp.asarray(x))
        assert array_equal(got, xp.asarray(expect))
        assert got.ndim >= 2

    def test_atleast_batched_multiple_args(self) -> None:
        """Test the `atleast_batched` function with multiple arguments."""
        # ArrayLike
        x = xp.asarray([1, 2, 3])
        y = xp.asarray([4, 5, 6])
        result = atleast_batched(x, y)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert array_equal(result[0], x[:, None])
        assert array_equal(result[1], y[:, None])

        # Quantity
        x = Quantity(x, "m")
        y = Quantity(y, "m")
        result = atleast_batched(x, y)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Quantity)
        assert isinstance(result[1], Quantity)
        assert array_equal(result[0], Quantity(x.value[:, None], "m"))
        assert array_equal(result[1], Quantity(y.value[:, None], "m"))


class TestBatchedShape:
    """Test the `galax.utils._shape.batched_shape` function."""

    @pytest.mark.parametrize(
        ("arr", "expect_ndim", "expect"),
        [
            # ArrayLike
            (xp.asarray(42), 0, ((), ())),
            (xp.asarray([1]), 0, ((1,), ())),
            (xp.asarray([1, 2, 3]), 1, ((), (3,))),
            (xp.asarray([[1, 2], [3, 4]]), 1, ((2,), (2,))),
            (xp.asarray([[1, 2], [3, 4]]), 2, ((), (2, 2))),
            # Quantity
            (Quantity(42, "m"), 0, ((), ())),
            (Quantity([1], "m"), 0, ((1,), ())),
            (Quantity([1, 2, 3], "m"), 1, ((), (3,))),
            (Quantity([[1, 2], [3, 4]], "m"), 1, ((2,), (2,))),
            (Quantity([[1, 2], [3, 4]], "m"), 2, ((), (2, 2))),
        ],
    )
    def test_batched_shape(
        self,
        arr: jax.Array,
        expect_ndim: int,
        expect: tuple[tuple[int, ...], tuple[int, ...]],
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
            (xp.asarray(1), 0, xp.asarray(1)),
            (xp.asarray([2]), 0, xp.asarray([2])),
            (xp.asarray([1, 2]), 0, xp.asarray([1, 2])),
            (xp.asarray(1), 1, xp.asarray([1])),
            (xp.asarray([2]), 1, xp.asarray([[2]])),
            (xp.asarray([1, 2]), 1, xp.asarray([[1, 2]])),
            (xp.asarray(1), 2, xp.asarray([[1]])),
            # Quantity
            (Quantity(1, "m"), 0, Quantity(1, "m")),
            (Quantity([2], "m"), 0, Quantity([2], "m")),
            (Quantity([1, 2], "m"), 0, Quantity([1, 2], "m")),
            (Quantity(1, "m"), 1, Quantity([1], "m")),
            (Quantity([2], "m"), 1, Quantity([[2]], "m")),
            (Quantity([1, 2], "m"), 1, Quantity([[1, 2]], "m")),
            (Quantity(1, "m"), 2, Quantity([[1]], "m")),
        ],
    )
    def test_expand_batch_dims(
        self, arr: jax.Array, ndim: int, expect: jax.Array
    ) -> None:
        """Test :func:`galax.utils._shape.expand_batch_dims`."""
        got = expand_batch_dims(arr, ndim=ndim)
        assert array_equal(got, expect)
        assert got.shape == expect.shape


class TestExpandArrDims:
    """Test :func:`galax.utils._shape.expand_arr_dims`."""

    @pytest.mark.parametrize(
        ("arr", "ndim", "expect"),
        [
            # ArrayLike
            (xp.asarray(1), 0, xp.asarray(1)),
            (xp.asarray([2]), 0, xp.asarray([2])),
            (xp.asarray([1, 2]), 0, xp.asarray([1, 2])),
            (xp.asarray(1), 1, xp.asarray([1])),
            (xp.asarray([2]), 1, xp.asarray([[2]])),
            (xp.asarray([1, 2]), 1, xp.asarray([[1], [2]])),
            (xp.asarray(1), 2, xp.asarray([[1]])),
            # Quantity
            (Quantity(1, "m"), 0, Quantity(1, "m")),
            (Quantity([2], "m"), 0, Quantity([2], "m")),
            (Quantity([1, 2], "m"), 0, Quantity([1, 2], "m")),
            (Quantity(1, "m"), 1, Quantity([1], "m")),
            (Quantity([2], "m"), 1, Quantity([[2]], "m")),
            (Quantity([1, 2], "m"), 1, Quantity([[1], [2]], "m")),
            (Quantity(1, "m"), 2, Quantity([[1]], "m")),
        ],
    )
    def test_expand_arr_dims(
        self, arr: jax.Array, ndim: int, expect: jax.Array
    ) -> None:
        """Test :func:`galax.utils._shape.expand_arr_dims`."""
        got = expand_arr_dims(arr, ndim=ndim)
        assert array_equal(got, expect)
        assert got.shape == expect.shape

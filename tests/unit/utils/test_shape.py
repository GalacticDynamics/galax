"""Test the `galax.utils._shape` module."""

import re

import jax.experimental.array_api as xp
import jax.numpy as jnp
import pytest
from jax.numpy import array_equal

from galax.utils._shape import atleast_batched, batched_shape


class TestAtleastBatched:
    """Test the `atleast_batched` function."""

    def test_atleast_batched_no_args(self):
        """Test the `atleast_batched` function with no arguments."""
        with pytest.raises(
            ValueError,
            match=re.escape("atleast_batched() requires at least one argument"),
        ):
            _ = atleast_batched()

    def test_atleast_batched_example(self):
        """Test the `atleast_batched` function with an example."""
        x = xp.asarray([1, 2, 3])
        # `atleast_batched` versus `atleast_2d`
        assert array_equal(atleast_batched(x), x[:, None])
        assert array_equal(jnp.atleast_2d(x), x[None, :])

    @pytest.mark.parametrize(
        ("x", "expect"),
        [
            (0, [[0]]),  # scalar -> 2D array
            ([1], [[1]]),  # 1D array -> 2D array
            ([[1]], [[1]]),  # 2D array -> 2D array
            ([[[1]]], [[[1]]]),  # 3D array -> 3D array
            ([1, 2, 3], [[1], [2], [3]]),
        ],
    )
    def test_atleast_batched_one_arg(self, x, expect):
        """Test the `atleast_batched` function with one argument."""
        got = atleast_batched(xp.asarray(x))
        assert array_equal(got, xp.asarray(expect))
        assert got.ndim >= 2

    def test_atleast_batched_multiple_args(self):
        """Test the `atleast_batched` function with multiple arguments."""
        x = xp.asarray([1, 2, 3])
        y = xp.asarray([4, 5, 6])
        result = atleast_batched(x, y)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert array_equal(result[0], x[:, None])
        assert array_equal(result[1], y[:, None])


class TestBatchedShape:
    """Test the `galax.utils._shape.batched_shape` function."""

    @pytest.mark.parametrize(
        ("arr", "expect_ndim", "expect"),
        [
            (xp.asarray(42), 0, ((), ())),
            (xp.asarray([1]), 0, ((1,), ())),
            (xp.asarray([1, 2, 3]), 1, ((), (3,))),
            (xp.asarray([[1, 2], [3, 4]]), 1, ((2,), (2,))),
            (xp.asarray([[1, 2], [3, 4]]), 2, ((), (2, 2))),
        ],
    )
    def test_batched_shape(self, arr, expect_ndim, expect):
        """Test the `galax.utils._shape.batched_shape` function."""
        batch, shape = batched_shape(arr, expect_ndim=expect_ndim)
        assert batch == expect[0]
        assert shape == expect[1]

"""Test the `galax.utils._shape` module."""

import re

import jax.numpy as xp
import pytest

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
        x = xp.array([1, 2, 3])
        # `atleast_batched` versus `atleast_2d`
        assert xp.array_equal(atleast_batched(x), x[:, None])
        assert xp.array_equal(xp.atleast_2d(x), x[None, :])

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
        got = atleast_batched(xp.array(x))
        assert xp.array_equal(got, xp.array(expect))
        assert got.ndim >= 2

    def test_atleast_batched_multiple_args(self):
        """Test the `atleast_batched` function with multiple arguments."""
        x = xp.array([1, 2, 3])
        y = xp.array([4, 5, 6])
        result = atleast_batched(x, y)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert xp.array_equal(result[0], x[:, None])
        assert xp.array_equal(result[1], y[:, None])


class TestBatchedShape:
    """Test the `galax.utils._shape.batched_shape` function."""

    @pytest.mark.parametrize(
        ("arr", "expect_ndim", "expect"),
        [
            (xp.array(42), 0, ((), ())),
            (xp.array([1]), 0, ((1,), ())),
            (xp.array([1, 2, 3]), 1, ((), (3,))),
            (xp.array([[1, 2], [3, 4]]), 1, ((2,), (2,))),
            (xp.array([[1, 2], [3, 4]]), 2, ((), (2, 2))),
        ],
    )
    def test_batched_shape(self, arr, expect_ndim, expect):
        """Test the `galax.utils._shape.batched_shape` function."""
        batch, shape = batched_shape(arr, expect_ndim=expect_ndim)
        assert batch == expect[0]
        assert shape == expect[1]

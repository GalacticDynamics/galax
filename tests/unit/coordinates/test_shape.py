"""Test the `galax.coordinates._src.shape` module."""

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates.custom_types as gt
from galax.coordinates._src.shape import batched_shape


class TestBatchedShape:
    """Test the `galax.coordinates._src.shape.batched_shape` function."""

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
            (u.Q(42, "m"), 0, ((), ())),
            (u.Q([1], "m"), 0, ((1,), ())),
            (u.Q([1, 2, 3], "m"), 1, ((), (3,))),
            (u.Q([[1, 2], [3, 4]], "m"), 1, ((2,), (2,))),
            (u.Q([[1, 2], [3, 4]], "m"), 2, ((), (2, 2))),
        ],
    )
    def test_batched_shape(
        self, arr: jax.Array, expect_ndim: int, expect: tuple[gt.Shape, gt.Shape]
    ) -> None:
        """Test the `galax.coordinates._src.shape.batched_shape` function."""
        batch, shape = batched_shape(arr, expect_ndim=expect_ndim)
        assert batch == expect[0]
        assert shape == expect[1]

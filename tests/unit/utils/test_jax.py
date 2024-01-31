"""Test the :mod:`galax.utils._jax` module."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from galax.utils._jax import vectorize_method


def test_vectorize_method() -> None:
    """Test the vectorize_method function."""

    class A:
        def __init__(self, x):
            self.x = x

        @vectorize_method(signature="(3)->()")
        def func(self, y: Float[Array, "batch N"]) -> Float[Array, "batch"]:
            return self.x + jnp.sum(y)

    a = A(1)
    assert a.func(jnp.array([1, 2, 3])) == 7

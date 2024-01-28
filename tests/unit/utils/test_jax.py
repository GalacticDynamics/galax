import jax.numpy as jnp
from jaxtyping import Array, Float

from galax.utils import partial_vectorize, vectorize_method


def test_partial_vectorize():
    """Test the partial_vectorize function."""

    def func(x: Float[Array, "batch N"]) -> Float[Array, "batch"]:
        return jnp.sum(x)

    vectorize_func = partial_vectorize(signature="(3)->()")(func)
    assert vectorize_func(jnp.array([1, 2, 3])) == 6

    # The real test is comparing this to the output of `jax.vectorize`.
    x = jnp.array([1, 2, 3])
    assert vectorize_func(x) == jnp.vectorize(func, signature="(3)->()")(x)


def test_vectorize_method():
    """Test the vectorize_method function."""

    class A:
        def __init__(self, x):
            self.x = x

        @vectorize_method(signature="(3)->()")
        def func(self, y: Float[Array, "batch N"]) -> Float[Array, "batch"]:
            return self.x + jnp.sum(y)

    a = A(1)
    assert a.func(jnp.array([1, 2, 3])) == 7

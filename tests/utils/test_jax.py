import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from galdynamix.utils import (
    partial_jit,
    partial_vectorize,
    partial_vmap,
    vectorize_method,
)


def test_partial_jit():
    """Test the partial_jit function."""

    def func(x, y):
        return x + y

    jit_func = partial_jit()(func)
    assert jit_func(1, 2) == 3

    # The real test is comparing this to the output of `jax.jit`.
    assert jit_func(1, 2) == jax.jit(func)(1, 2)

    # TODO: test all the kwarg options.


def test_partial_vmap():
    """Test the partial_vmap function."""

    def func(x: Float[Array, "batch N"]) -> Float[Array, "batch"]:
        return jnp.sum(x)

    vmap_func = partial_vmap(in_axes=0)(func)
    x = jnp.array([[1, 2, 3]])
    assert vmap_func(x) == 6

    # The real test is comparing this to the output of `jax.vmap`.
    assert vmap_func(x) == jax.vmap(func, in_axes=0)(x)


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

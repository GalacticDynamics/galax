"""Utility Functions."""
# ruff:noqa: UP037

from __future__ import annotations

from jax.scipy.special import gamma
from jaxtyping import Array, Integer


def factorial(n: Integer[Array, "1"]) -> Integer[Array, "1"]:
    """Factorial helper function."""
    return gamma(n + 1.0)  # n! = gamma(n+1)

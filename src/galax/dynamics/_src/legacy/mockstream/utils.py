"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from functools import partial
from typing import Any, TypeVar, cast

import jax
import optype as op
from jaxtyping import Array, Bool
from quax import quaxify

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


def _identity(x: T) -> T:
    return x


@partial(quaxify)  # TODO: move this `quaxify` up the function call stack
def _reverse(x: op.CanGetitem[Any, T]) -> T:
    return x[::-1]


def cond_reverse(pred: Bool[Array, ""], x: T) -> T:
    return cast(T, jax.lax.cond(pred, _reverse, _identity, x))

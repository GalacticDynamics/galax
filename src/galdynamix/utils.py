"""galdynamix: Galactic Dynamix in Jax"""

from __future__ import annotations

__all__: list[str] = ["jit_method"]

from functools import partial
from typing import Any, Callable, TypeVar

import jax
from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


def jit_method(
    **kwargs: Any,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    return partial(jax.jit, static_argnums=(0,), **kwargs)

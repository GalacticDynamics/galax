"""galdynamix: Galactic Dynamix in Jax"""

from __future__ import annotations

__all__ = ["partial_jit"]

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import (
    Any,
    NotRequired,
    TypedDict,
    TypeVar,
)

import jax
from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


class JITKwargs(TypedDict):
    in_shardings: NotRequired[Any]
    out_shardings: NotRequired[Any]
    static_argnums: NotRequired[int | Sequence[int] | None]
    static_argnames: NotRequired[str | Iterable[str] | None]
    donate_argnums: NotRequired[int | Sequence[int] | None]
    donate_argnames: NotRequired[str | Iterable[str] | None]
    keep_unused: NotRequired[bool]
    device: NotRequired[jax.Device | None]
    backend: NotRequired[str | None]
    inline: NotRequired[bool]


def partial_jit(
    **kwargs: Any,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    return partial(jax.jit, **kwargs)

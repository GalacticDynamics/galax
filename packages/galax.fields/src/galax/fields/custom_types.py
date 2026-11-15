"""Type hints for `galax.fields`. Private API.

As indicated by `__all__`, this module does not export any names. The type hints
defined here may be changed or removed without notice.

Notes
-----
- "Sz<X>" stands for the shape, which is the primary (not batch) shape.

"""

__all__: tuple[str, ...] = ()

from jaxtyping import Array, ArrayLike, PyTree, Real, ScalarLike
from typing import TypeAlias

Sz0: TypeAlias = Real[Array, ""]
LikeSz0: TypeAlias = Real[ArrayLike, ""]
SzAny: TypeAlias = Real[Array, "..."]
RealScalarLike: TypeAlias = Real[ScalarLike, ""]
SzTime: TypeAlias = Real[Array, "time"]
Terms: TypeAlias = PyTree

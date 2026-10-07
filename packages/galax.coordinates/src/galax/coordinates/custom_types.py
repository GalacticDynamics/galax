"""Type hints for `galax.coordinates`. Private API.

As indicated by `__all__`, this module does not export any names. The type hints
defined here may be changed or removed without notice.

Notes
-----
- "Bt" stands for "batch", which in `jaxtyping` is '#batch'.
- "BBt" stands for broadcast batchable, which in `jaxtyping` is '*#batch'.
- "Sz<X>" stands for the shape, which is the primary (not batch) shape.
  For example, "Sz3" is a 3-vector and "Sz33" is a 3x3 matrix.
- "Qu" stands for `unxt.quantity.AbstractQuantity`.

"""

__all__: tuple[str, ...] = ()

from jaxtyping import Array, Float, Real, Shaped
from typing import TypeAlias

import unxt as u

Shape: TypeAlias = tuple[int, ...]
QuSz0: TypeAlias = Real[u.AbstractQuantity, ""]
BBtQuSz0: TypeAlias = Real[QuSz0, "*#batch"]
FloatQuSz0: TypeAlias = Float[u.AbstractQuantity, ""]
BtFloatQuSz0: TypeAlias = Float[FloatQuSz0, "*batch"]
BBtFloatQuSz0: TypeAlias = Float[FloatQuSz0, "*#batch"]
Sz6: TypeAlias = Real[Array, "6"]
BtSz6: TypeAlias = Real[Sz6, "*batch"]
BBtSz7: TypeAlias = Real[Array, "*#batch 7"]
SzN: TypeAlias = Shaped[Array, "N"]
QuSzTime: TypeAlias = Real[u.AbstractQuantity, "time"]

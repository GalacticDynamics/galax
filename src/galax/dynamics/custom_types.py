"""Type hints for `galax.dynamics`. Private API.

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

from jaxtyping import Array, Int, Real, ScalarLike
from typing import Any, TypeAlias

import unxt as u

from galax.coordinates.custom_types import (
    BBtFloatQuSz0 as BBtFloatQuSz0,
    BBtQuSz0 as BBtQuSz0,
    BBtSz7 as BBtSz7,
    BtFloatQuSz0 as BtFloatQuSz0,
    BtSz6 as BtSz6,
    FloatQuSz0 as FloatQuSz0,
    QuSz0 as QuSz0,
    QuSzTime as QuSzTime,
    Shape as Shape,
    Sz6 as Sz6,
    SzN as SzN,
)
from galax.potential.custom_types import (
    BBtLikeSz0 as BBtLikeSz0,
    BBtQorVSz0 as BBtQorVSz0,
    BBtQorVSz3 as BBtQorVSz3,
    BBtQuSz3 as BBtQuSz3,
    BBtQuSz4 as BBtQuSz4,
    BBtSz0 as BBtSz0,
    BBtSz3 as BBtSz3,
    BtQuSz0 as BtQuSz0,
    BtSz3 as BtSz3,
    FloatQuSz3 as FloatQuSz3,
    FloatSz0 as FloatSz0,
    FloatSz3 as FloatSz3,
    LikeSz0 as LikeSz0,
    QuSz3 as QuSz3,
    Sz0 as Sz0,
    Sz3 as Sz3,
    Sz33 as Sz33,
    SzAny as SzAny,
)

OptArgs: TypeAlias = dict[str, Any] | None
IntSz0: TypeAlias = Int[Array, ""]
RealScalarLike: TypeAlias = Real[ScalarLike, ""]
QuSz1: TypeAlias = Real[u.AbstractQuantity, "1"]
BtQuSz3: TypeAlias = Real[QuSz3, "*batch"]
BBtSz4: TypeAlias = Real[Array, "*#batch 4"]
BBtSz6: TypeAlias = Real[Sz6, "*#batch"]
SzTime: TypeAlias = Real[Array, "time"]

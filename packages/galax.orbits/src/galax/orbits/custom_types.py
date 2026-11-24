"""Type hints for `galax.orbits`.

As indicated by `__all__`, this module does not export any names. The type hints
defined here may be changed or removed without notice.

Notes
-----
- "Bt" stands for "batch", which in `jaxtyping` is '#batch'.
- "BBt" stands for broadcast batchable, which in `jaxtyping` is '*#batch'.
- "Sz<X>" stands for the shape, which is the primary (not batch) shape.
  For example, "Sz3" is a 3-vector and "Sz33" is a 3x3 matrix.
- "Qu" stands for `unxt.quantity.AbstractQuantity`.
- "Q", "P" and "A" are position, momentum (velocity) and acceleration; a
  trailing "arr" marks a plain array rather than a `unxt.Quantity`.

"""

__all__: tuple[str, ...] = ()

from jaxtyping import Array, Int, Real
from typing import Any, TypeAlias

import unxt as u

from galax.coordinates.custom_types import (
    BBtSz7 as BBtSz7,
    BtFloatQuSz0 as BtFloatQuSz0,
    BtSz6 as BtSz6,
    QuSz0 as QuSz0,
    QuSzTime as QuSzTime,
    Shape as Shape,
    Sz6 as Sz6,
)
from galax.potential.custom_types import (
    BBtLikeSz0 as BBtLikeSz0,
    BBtQuSz0 as BBtQuSz0,
    BBtQuSz3 as BBtQuSz3,
    BBtQuSz4 as BBtQuSz4,
    BBtSz0 as BBtSz0,
    BBtSz3 as BBtSz3,
    BtSz3 as BtSz3,
    FloatSz0 as FloatSz0,
    LikeSz0 as LikeSz0,
    QuSz3 as QuSz3,
    Sz0 as Sz0,
    Sz3 as Sz3,
)

OptArgs: TypeAlias = dict[str, Any] | None
QuSz1: TypeAlias = Real[u.AbstractQuantity, "1"]
BBtSz4: TypeAlias = Real[Array, "*#batch 4"]
BBtSz6: TypeAlias = Real[Sz6, "*#batch"]
IntSz0: TypeAlias = Int[Array, ""]
SzTime: TypeAlias = Real[Array, "time"]
BtQuSz3: TypeAlias = Real[QuSz3, "*batch"]

# ---------------------------
# Q, P, A

Qarr: TypeAlias = Sz3
Parr: TypeAlias = Sz3
BBtQarr: TypeAlias = BBtSz3
BBtQ: TypeAlias = BBtQuSz3
BtQarr: TypeAlias = BtSz3
BtQ: TypeAlias = BtQuSz3
BBtParr: TypeAlias = BBtSz3
BBtP: TypeAlias = BBtQuSz3
BtParr: TypeAlias = BtSz3
BtP: TypeAlias = BtQuSz3
BtAarr: TypeAlias = BtSz3

# ---------------------------
# QP, PA

BBtQParr: TypeAlias = tuple[BBtQarr, BBtParr]
BtQParr: TypeAlias = tuple[BtQarr, BtParr]
BtQP: TypeAlias = tuple[BtQ, BtP]
BBtQP: TypeAlias = tuple[BBtQ, BBtP]
QParr: TypeAlias = tuple[Qarr, Parr]
BtPAarr: TypeAlias = tuple[BtSz3, BtAarr]

"""Type hints for `galax.potential`. Private API.

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

__all__: list[str] = []

from typing import TypeAlias

import astropy.units as apyu
from jaxtyping import Array, ArrayLike, Float, Real, ScalarLike

import unxt as u

from galax.coordinates.custom_types import (
    BBtFloatQuSz0 as BBtFloatQuSz0,
    BBtQuSz0 as BBtQuSz0,
    BtFloatQuSz0 as BtFloatQuSz0,
    QuSz0 as QuSz0,
    Sz6 as Sz6,
    SzN as SzN,
)

Unit: TypeAlias = apyu.Unit | apyu.UnitBase | apyu.CompositeUnit
Sz0: TypeAlias = Real[Array, ""]
BtSz0: TypeAlias = Real[Sz0, "*batch"]
BBtSz0: TypeAlias = Real[Sz0, "*#batch"]
BtQuSz0: TypeAlias = Real[QuSz0, "*batch"]
BBtQorVSz0: TypeAlias = BBtQuSz0 | BBtSz0
FloatSz0: TypeAlias = Float[Array, ""]
BtFloatSz0: TypeAlias = Float[FloatSz0, "*batch"]
BBtFloatSz0: TypeAlias = Float[FloatSz0, "*#batch"]
LikeSz0: TypeAlias = Real[ArrayLike, ""]
BBtLikeSz0: TypeAlias = Real[LikeSz0, "*#batch"]
FloatSz3: TypeAlias = Float[Array, "3"]
FloatQuSz3: TypeAlias = Float[u.AbstractQuantity, "3"]
Sz3: TypeAlias = Real[Array, "3"]
BtSz3: TypeAlias = Real[Sz3, "*batch"]
BBtSz3: TypeAlias = Real[Sz3, "*#batch"]
QuSz3: TypeAlias = Real[u.AbstractQuantity, "3"]
BBtQuSz3: TypeAlias = Real[QuSz3, "*#batch"]
BBtQorVSz3: TypeAlias = BBtQuSz3 | BBtSz3
BBtLikeSz4: TypeAlias = Real[ArrayLike, "*#batch 4"]
BBtQuSz4: TypeAlias = Real[u.AbstractQuantity, "*#batch 4"]
Sz33: TypeAlias = Real[Array, "3 3"]
BBtSz33: TypeAlias = Real[Sz33, "*#batch"]
QuSz33: TypeAlias = Real[u.AbstractQuantity, "3 3"]
BBtQuSz33: TypeAlias = Real[QuSz33, "*#batch"]
SzAny: TypeAlias = Real[Array, "..."]
QuSzAny: TypeAlias = Real[u.AbstractQuantity, "..."]
XYZArrayLike: TypeAlias = (
    Real[ArrayLike, "*#batch 3"]
    | list[ScalarLike]
    | tuple[ScalarLike, ScalarLike, ScalarLike]
)
Params: TypeAlias = dict[str, Real[Array, "..."]]

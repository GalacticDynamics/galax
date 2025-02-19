"""Type hints for galax. Private API.

As indicated by `__all__`, this module does not export any names. The type hints
defined here may be changed or removed without notice. They are intended for use
in other modules within the `galax` package.

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
from jaxtyping import Array, ArrayLike, Float, Int, Real, Scalar, ScalarLike, Shaped

import unxt as u
from unxt.quantity import AbstractQuantity

# =============================================================================

Shape: TypeAlias = tuple[int, ...]
Dimension: TypeAlias = apyu.PhysicalType
Unit: TypeAlias = apyu.Unit | apyu.UnitBase | apyu.CompositeUnit


# =============================================================================
# TODO: sort

RealScalarLike: TypeAlias = Real[ScalarLike, ""]


# =============================================================================
# Vectors

# ---------------------------
# 0-scalar
# Any
Sz0: TypeAlias = Scalar
BtSz0: TypeAlias = Shaped[Sz0, "*batch"]
BBtSz0: TypeAlias = Shaped[Sz0, "*#batch"]
BBtQuSz0: TypeAlias = Shaped[AbstractQuantity, "*#batch"]

# Integer
IntSz0: TypeAlias = Int[Array, ""]
IntQuSz0: TypeAlias = Int[AbstractQuantity, ""]
IntLike: TypeAlias = IntSz0 | int

# Float
FloatSz0: TypeAlias = Float[Array, ""]
BtFloatSz0: TypeAlias = Shaped[FloatSz0, "*batch"]
BBtFloatSz0: TypeAlias = Shaped[FloatSz0, "*#batch"]

FloatQuSz0: TypeAlias = Float[AbstractQuantity, ""]
BtFloatQuSz0: TypeAlias = Shaped[FloatQuSz0, "*batch"]
BBtFloatQuSz0: TypeAlias = Shaped[FloatQuSz0, "*#batch"]

FloatLike: TypeAlias = FloatSz0 | float | int

# Real
RealSz0: TypeAlias = Real[Array, ""]
BtRealSz0: TypeAlias = Shaped[RealSz0, "*batch"]
BBtRealSz0: TypeAlias = Shaped[RealSz0, "*#batch"]

RealLikeSz0: TypeAlias = Real[ArrayLike, ""]
BtRealLikeSz0: TypeAlias = Shaped[RealLikeSz0, "*batch"]
BBtRealLikeSz0: TypeAlias = Shaped[RealLikeSz0, "*#batch"]

RealQuSz0: TypeAlias = Real[AbstractQuantity, ""]
BtRealQuSz0: TypeAlias = Shaped[RealQuSz0, "*batch"]
BBtRealQuSz0: TypeAlias = Shaped[RealQuSz0, "*#batch"]

RealSz0Like: TypeAlias = FloatLike | IntLike  # A float or int or float(/int) scalar.

# ---------------------------
# 1-vector
Sz1: TypeAlias = Shaped[Array, "1"]
QuSz1: TypeAlias = Shaped[AbstractQuantity, "1"]

# ---------------------------
# A 3-vector, e.g. q=(x, y, z) or p=(vx, vy, vz).
Sz3: TypeAlias = Shaped[Array, "3"]
BtSz3: TypeAlias = Shaped[Sz3, "*batch"]
BBtSz3: TypeAlias = Shaped[Sz3, "*#batch"]

QuSz3: TypeAlias = Shaped[AbstractQuantity, "3"]
BtQuSz3: TypeAlias = Shaped[QuSz3, "*batch"]
BBtQuSz3: TypeAlias = Shaped[QuSz3, "*#batch"]

FloatSz3: TypeAlias = Float[Array, "3"]
BtFloatSz3: TypeAlias = Float[FloatSz3, "*batch"]
FloatQuSz3: TypeAlias = Float[AbstractQuantity, "3"]
BtFloatQuSz3: TypeAlias = Float[QuSz3, "*batch"]
BBtFloatQuSz3: TypeAlias = Float[QuSz3, "*#batch"]

RealSz3: TypeAlias = Real[Array, "3"]
BtRealSz3: TypeAlias = Real[RealSz3, "*batch"]
BBtRealSz3: TypeAlias = Real[RealSz3, "*#batch"]

RealLikeSz3: TypeAlias = Real[ArrayLike, "3"]
BtRealLikeSz3: TypeAlias = Real[RealLikeSz3, "*batch"]
BBtRealLikeSz3: TypeAlias = Real[RealLikeSz3, "*#batch"]

RealQuSz3: TypeAlias = Real[AbstractQuantity, "3"]
BtRealQuSz3: TypeAlias = Shaped[RealQuSz3, "*batch"]
BBtRealQuSz3: TypeAlias = Shaped[RealQuSz3, "*#batch"]

# ---------------------------
# 4-vector

BBtSz4: TypeAlias = Shaped[Array, "*#batch 4"]
BBtRealLikeSz4: TypeAlias = Real[ArrayLike, "*#batch 4"]
BBtRealQuSz4: TypeAlias = Real[u.AbstractQuantity, "*#batch 4"]

# ---------------------------
# 6-vector
Sz6: TypeAlias = Shaped[Array, "6"]
BtSz6: TypeAlias = Shaped[Sz6, "*batch"]
BBtSz6: TypeAlias = Shaped[Sz6, "*#batch"]

# ---------------------------
# 7-vector
Sz7: TypeAlias = Shaped[Array, "7"]
BtSz7: TypeAlias = Shaped[Sz7, "*batch"]
BBtSz7: TypeAlias = Shaped[Sz7, "*#batch"]

# ---------------------------
# 3x3 matrix

Sz33: TypeAlias = Shaped[Array, "3 3"]
BBtSz33: TypeAlias = Shaped[Sz33, "*#batch"]
BBtRealSz33: TypeAlias = Real[Array, "*#batch 3 3"]
FloatSz33: TypeAlias = Float[Array, "3 3"]

QuSz33: TypeAlias = Shaped[AbstractQuantity, "3 3"]
BtQuSz33: TypeAlias = Shaped[QuSz33, "*batch"]
BBtQuSz33: TypeAlias = Shaped[QuSz33, "*#batch"]
FloatQuSz33: TypeAlias = Float[AbstractQuantity, "3 3"]
BBtRealQuSz33: TypeAlias = Real[AbstractQuantity, "*#batch 3 3"]

# ================================
# Specific shapes

# N-vector
SzN: TypeAlias = Shaped[Array, "N"]
"""An (N,)-vector."""

# Time vector
SzTime: TypeAlias = Shaped[Array, "time"]
QuSzTime: TypeAlias = Shaped[AbstractQuantity, "time"]

# A float array with any shape.
FloatSzAny: TypeAlias = Float[Array, "..."]
FloatQuSzAny: TypeAlias = Float[AbstractQuantity, "..."]

# ================================
# Specific Quantity types

MassSz0: TypeAlias = Shaped[u.Quantity["mass"], ""]
MassBtSz0: TypeAlias = Shaped[u.Quantity["mass"], "*batch"]
MassBBtSz0: TypeAlias = Shaped[u.Quantity["mass"], "*#batch"]

TimeSz0: TypeAlias = Shaped[u.Quantity["time"], ""]
TimeBtSz0: TypeAlias = Shaped[u.Quantity["time"], "*batch"]
TimeBBtSz0: TypeAlias = Shaped[u.Quantity["time"], "*#batch"]

# ===============================
# TODO: sort


XYZArrayLike: TypeAlias = (
    BBtRealLikeSz3 | list[ScalarLike] | tuple[ScalarLike, ScalarLike, ScalarLike]
)

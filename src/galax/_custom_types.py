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
QuSz0: TypeAlias = Shaped[AbstractQuantity, ""]
BtQuSz0: TypeAlias = Shaped[QuSz0, "*batch"]
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

QuSz1: TypeAlias = Real[AbstractQuantity, "1"]

# ---------------------------
# A 3-vector, e.g. q=(x, y, z) or p=(vx, vy, vz).

FloatSz3: TypeAlias = Float[Array, "3"]
FloatQuSz3: TypeAlias = Float[AbstractQuantity, "3"]

Sz3: TypeAlias = Real[Array, "3"]
BtSz3: TypeAlias = Real[Sz3, "*batch"]
BBtSz3: TypeAlias = Real[Sz3, "*#batch"]

QuSz3: TypeAlias = Real[AbstractQuantity, "3"]
BtQuSz3: TypeAlias = Real[QuSz3, "*batch"]
BBtQuSz3: TypeAlias = Real[QuSz3, "*#batch"]

# ---------------------------
# 4-vector

BBtSz4: TypeAlias = Real[Array, "*#batch 4"]
BBtLikeSz4: TypeAlias = Real[ArrayLike, "*#batch 4"]
BBtQuSz4: TypeAlias = Real[u.AbstractQuantity, "*#batch 4"]

# ---------------------------
# 6-vector

Sz6: TypeAlias = Real[Array, "6"]
BtSz6: TypeAlias = Real[Sz6, "*batch"]
BBtSz6: TypeAlias = Real[Sz6, "*#batch"]

# ---------------------------
# 7-vector

BBtSz7: TypeAlias = Real[Array, "*#batch 7"]

# ---------------------------
# 3x3 matrix

Sz33: TypeAlias = Real[Array, "3 3"]
BBtSz33: TypeAlias = Real[Sz33, "*#batch"]

QuSz33: TypeAlias = Real[AbstractQuantity, "3 3"]
BBtQuSz33: TypeAlias = Real[QuSz33, "*#batch"]

# ================================
# Specific shapes

# N-vector
SzN: TypeAlias = Shaped[Array, "N"]
"""An (N,)-vector."""

# Time vector
SzTime: TypeAlias = Real[Array, "time"]
QuSzTime: TypeAlias = Real[AbstractQuantity, "time"]

# A real array with any shape.
SzAny: TypeAlias = Real[Array, "..."]
QuSzAny: TypeAlias = Real[AbstractQuantity, "..."]

# ===============================
# TODO: sort


XYZArrayLike: TypeAlias = (
    Real[ArrayLike, "*#batch 3"]
    | list[ScalarLike]
    | tuple[ScalarLike, ScalarLike, ScalarLike]
)

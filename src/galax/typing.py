"""Type hints for galax.

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
from jaxtyping import Array, Float, Int, Real, Scalar, Shaped

import unxt as u
from unxt.quantity import AbstractQuantity

# =============================================================================

Shape: TypeAlias = tuple[int, ...]
Dimension: TypeAlias = apyu.PhysicalType
Unit: TypeAlias = apyu.Unit | apyu.UnitBase | apyu.CompositeUnit

# =============================================================================
# Vectors

# Any
BBtScalarSz0: TypeAlias = Shaped[Scalar, "*#batch"]

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
BBtRealSz0: TypeAlias = Shaped[RealSz0, "*#batch"]

RealQuSz0: TypeAlias = Real[AbstractQuantity, ""]
BtRealQuSz0: TypeAlias = Shaped[RealQuSz0, "*batch"]
BBtRealQuSz0: TypeAlias = Shaped[RealQuSz0, "*#batch"]


RealSz0Like: TypeAlias = FloatLike | IntLike  # A float or int or float(/int) scalar.

# 1-vector
Sz1: TypeAlias = Float[Array, "1"]
QuSz1: TypeAlias = Float[AbstractQuantity, "1"]

# A 3-vector, e.g. q=(x, y, z) or p=(vx, vy, vz).
Sz3: TypeAlias = Float[Array, "3"]
BtSz3: TypeAlias = Shaped[Sz3, "*batch"]

QuSz3: TypeAlias = Float[AbstractQuantity, "3"]
BtQuSz3: TypeAlias = Shaped[QuSz3, "*batch"]
BtFloatQuSz3: TypeAlias = Float[QuSz3, "*batch"]

# 6-vector
Sz6: TypeAlias = Float[Array, "6"]
BtSz6: TypeAlias = Shaped[Sz6, "*batch"]
BBtSz6: TypeAlias = Shaped[Sz6, "*#batch"]

# 7-vector
Sz7: TypeAlias = Float[Array, "7"]
BtSz7: TypeAlias = Shaped[Sz7, "*batch"]
BBtSz7: TypeAlias = Shaped[Sz7, "*#batch"]

# 3x3 matrix
QuSz33: TypeAlias = Float[AbstractQuantity, "3 3"]
BtQuSz33: TypeAlias = Shaped[QuSz33, "*batch"]

# ================================
# Specific shapes

# N-vector
SzN: TypeAlias = Float[Array, "N"]
"""An (N,)-vector."""

# Time vector
SzTime: TypeAlias = Float[Array, "time"]
QuSzTime: TypeAlias = Float[AbstractQuantity, "time"]

# A float array with any shape.
FloatQuSzAny: TypeAlias = Float[AbstractQuantity, "..."]

# ================================
# Specific array types

MassSz0: TypeAlias = Shaped[u.Quantity["mass"], ""]
MassBtSz0: TypeAlias = Shaped[u.Quantity["mass"], "*batch"]
MassBBtSz0: TypeAlias = Shaped[u.Quantity["mass"], "*#batch"]

TimeSz0: TypeAlias = Shaped[u.Quantity["time"], ""]
TimeBtSz0: TypeAlias = Shaped[u.Quantity["time"], "*batch"]
TimeBBtSz0: TypeAlias = Shaped[u.Quantity["time"], "*#batch"]

LengthSz3: TypeAlias = Shaped[u.Quantity["length"], "3"]
LengthBtSz3: TypeAlias = Shaped[LengthSz3, "*batch"]
LengthBBtSz3: TypeAlias = Shaped[LengthSz3, "*#batch"]

SpeedSz3: TypeAlias = Shaped[u.Quantity["speed"], "3"]
SpeedBtSz3: TypeAlias = Shaped[SpeedSz3, "*batch"]
SpeedBBtSz3: TypeAlias = Shaped[SpeedSz3, "*#batch"]

SpecificEnergySz0: TypeAlias = Float[u.Quantity["specific_energy"], ""]
SpecificEnergyBtSz0: TypeAlias = Float[u.Quantity["specific_energy"], "*batch"]

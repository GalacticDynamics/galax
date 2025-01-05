"""Type hints for galax.

As indicated by `__all__`, this module does not export any names. The type hints
defined here may be changed or removed without notice. They are intended for use
in other modules within the `galax` package.

"""

__all__: list[str] = []

from typing import TypeAlias

import astropy.units as apyu
from jaxtyping import Array, Float, Integer, Shaped

import unxt as u
from unxt.quantity import AbstractQuantity

# =============================================================================

Shape: TypeAlias = tuple[int, ...]
Dimension: TypeAlias = apyu.PhysicalType
Unit: TypeAlias = apyu.Unit | apyu.UnitBase | apyu.CompositeUnit

# =============================================================================
# Scalars

# An integer scalar.
IntScalar: TypeAlias = Integer[Array, ""]
IntQScalar: TypeAlias = Integer[AbstractQuantity, ""]
IntLike: TypeAlias = IntScalar | int

# A float scalar.
FloatScalar: TypeAlias = Float[Array, ""]
FloatQScalar: TypeAlias = Float[AbstractQuantity, ""]

FloatLike: TypeAlias = FloatScalar | float | int
"""A float(/int) or float scalar."""

# A float or integer scalar.
RealScalar: TypeAlias = FloatScalar | IntScalar
RealQScalar: TypeAlias = FloatQScalar | IntQScalar

# A float or integer or float(/int) scalar.
RealScalarLike: TypeAlias = FloatLike | IntLike


# =============================================================================
# Vectors

# -----------------------------------------------------------------------------
# Shaped

# 1-vector
Vec1: TypeAlias = Float[Array, "1"]
QVec1: TypeAlias = Float[AbstractQuantity, "1"]

# A 3-vector, e.g. q=(x, y, z) or p=(vx, vy, vz).
Vec3: TypeAlias = Float[Array, "3"]
QVec3: TypeAlias = Float[AbstractQuantity, "3"]

Vec6: TypeAlias = Float[Array, "6"]
"""A 6-vector e.g. w=(x, y, z, vx, vy, vz)."""

Vec7: TypeAlias = Float[Array, "7"]
"""A 7-vector e.g. w=(x, y, z, vx, vy, vz, t)."""

VecN: TypeAlias = Float[Array, "N"]
"""An (N,)-vector."""

# -----------------
# Specific

# Time vector
VecTime: TypeAlias = Float[Array, "time"]
QVecTime: TypeAlias = Float[AbstractQuantity, "time"]

# -----------------------------------------------------------------------------
# Vector Batches

# -----------------
# Scalars

BatchFloatScalar: TypeAlias = Shaped[FloatScalar, "*batch"]

BatchableFloatQScalar: TypeAlias = Shaped[FloatQScalar, "*#batch"]

BatchFloatQScalar: TypeAlias = Shaped[FloatQScalar, "*batch"]

BatchRealQScalar: TypeAlias = Shaped[RealQScalar, "*batch"]
BatchableRealScalar: TypeAlias = Shaped[RealScalar, "*#batch"]
BatchableRealQScalar: TypeAlias = Shaped[RealQScalar, "*#batch"]

# -----------------
# Batched

# Zero or more batches of 3-vectors.
BatchVec3: TypeAlias = Shaped[Vec3, "*batch"]
BatchQVec3: TypeAlias = Shaped[QVec3, "*batch"]

# Zero or more batches of 6-vectors
BatchVec6: TypeAlias = Shaped[Vec6, "*batch"]
BatchableVec6: TypeAlias = Shaped[Vec6, "*#batch"]

# Zero or more batches of 7-vectors
BatchVec7: TypeAlias = Shaped[Vec7, "*batch"]

# -----------------
# Any Shape

# A float array with any shape.
FloatQAnyShape: TypeAlias = Float[AbstractQuantity, "..."]


# =============================================================================
# Matrices

QMatrix33: TypeAlias = Float[AbstractQuantity, "3 3"]
BatchQMatrix33: TypeAlias = Shaped[QMatrix33, "*batch"]


# =============================================================================

MassScalar: TypeAlias = Shaped[u.Quantity["mass"], ""]
MassBatchScalar: TypeAlias = Shaped[u.Quantity["mass"], "*batch"]
MassBatchableScalar: TypeAlias = Shaped[u.Quantity["mass"], "*#batch"]

TimeScalar: TypeAlias = Shaped[u.Quantity["time"], ""]
TimeBatchScalar: TypeAlias = Shaped[u.Quantity["time"], "*batch"]
TimeBatchableScalar: TypeAlias = Shaped[u.Quantity["time"], "*#batch"]

LengthVec3: TypeAlias = Shaped[u.Quantity["length"], "3"]
LengthBatchVec3: TypeAlias = Shaped[LengthVec3, "*batch"]
LengthBatchableVec3: TypeAlias = Shaped[LengthVec3, "*#batch"]

SpeedVec3: TypeAlias = Shaped[u.Quantity["speed"], "3"]
SpeedBatchVec3: TypeAlias = Shaped[SpeedVec3, "*batch"]
SpeedBatchableVec3: TypeAlias = Shaped[SpeedVec3, "*#batch"]

SpecificEnergyScalar: TypeAlias = Float[u.Quantity["specific_energy"], ""]
SpecificEnergyBatchScalar: TypeAlias = Float[u.Quantity["specific_energy"], "*batch"]


# =============================================================================

Qarr: TypeAlias = Vec3
BatchQarr: TypeAlias = Shaped[Qarr, "*batch"]

Parr: TypeAlias = Vec3
BatchParr: TypeAlias = Shaped[Parr, "*batch"]

BatchQParr: TypeAlias = tuple[BatchQarr, BatchParr]

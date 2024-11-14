"""Type hints for galax."""

__all__: list[str] = []

from typing import TypeAlias

import astropy.units as u
from jaxtyping import Array, Float, Integer, Shaped

from unxt.quantity import AbstractQuantity, Quantity

# =============================================================================

Shape: TypeAlias = tuple[int, ...]
Dimension: TypeAlias = u.PhysicalType
Unit: TypeAlias = u.Unit | u.UnitBase | u.CompositeUnit

# =============================================================================
# Scalars

# An integer scalar.
IntScalar = Integer[Array, ""]
IntQScalar = Integer[AbstractQuantity, ""]
IntLike = IntScalar | int

# A float scalar.
FloatScalar = Float[Array, ""]
FloatQScalar = Float[AbstractQuantity, ""]

FloatLike = FloatScalar | float | int
"""A float(/int) or float scalar."""

# A float or integer scalar.
RealScalar = FloatScalar | IntScalar
RealQScalar = FloatQScalar | IntQScalar

# A float or integer or float(/int) scalar.
RealScalarLike = FloatLike | IntLike


# =============================================================================
# Vectors

# -----------------------------------------------------------------------------
# Shaped

# 1-vector
Vec1 = Float[Array, "1"]
QVec1 = Float[AbstractQuantity, "1"]

# A 3-vector, e.g. q=(x, y, z) or p=(vx, vy, vz).
Vec3 = Float[Array, "3"]
QVec3 = Float[AbstractQuantity, "3"]

Vec6 = Float[Array, "6"]
"""A 6-vector e.g. w=(x, y, z, vx, vy, vz)."""

Vec7 = Float[Array, "7"]
"""A 7-vector e.g. w=(x, y, z, vx, vy, vz, t)."""

VecN = Float[Array, "N"]
"""An (N,)-vector."""

# -----------------
# Specific

# Time vector
VecTime = Float[Array, "time"]
QVecTime = Float[AbstractQuantity, "time"]

# -----------------------------------------------------------------------------
# Vector Batches

# -----------------
# Scalars

BatchFloatScalar = Shaped[FloatScalar, "*batch"]

BatchableFloatQScalar = Shaped[FloatQScalar, "*#batch"]

BatchFloatQScalar = Shaped[FloatQScalar, "*batch"]

BatchRealQScalar = Shaped[RealQScalar, "*batch"]
BatchableRealScalar = Shaped[RealScalar, "*#batch"]
BatchableRealQScalar = Shaped[RealQScalar, "*#batch"]

# -----------------
# Batched

# Zero or more batches of 3-vectors.
BatchVec3 = Shaped[Vec3, "*batch"]
BatchQVec3 = Shaped[QVec3, "*batch"]

# Zero or more batches of 6-vectors
BatchVec6 = Shaped[Vec6, "*batch"]
BatchableVec6 = Shaped[Vec6, "*#batch"]

# Zero or more batches of 7-vectors
BatchVec7 = Shaped[Vec7, "*batch"]

# -----------------
# Any Shape

# A float array with any shape.
FloatQAnyShape = Float[AbstractQuantity, "..."]


# =============================================================================
# Matrices

QMatrix33 = Float[AbstractQuantity, "3 3"]
BatchQMatrix33 = Shaped[QMatrix33, "*batch"]


# =============================================================================

MassScalar = Shaped[Quantity["mass"], ""]
MassBatchScalar = Shaped[Quantity["mass"], "*batch"]
MassBatchableScalar = Shaped[Quantity["mass"], "*#batch"]

TimeScalar = Shaped[Quantity["time"], ""]
TimeBatchScalar = Shaped[Quantity["time"], "*batch"]
TimeBatchableScalar = Shaped[Quantity["time"], "*#batch"]

LengthVec3 = Shaped[Quantity["length"], "3"]
LengthBatchVec3 = Shaped[LengthVec3, "*batch"]
LengthBatchableVec3 = Shaped[LengthVec3, "*#batch"]

SpeedVec3: TypeAlias = Shaped[Quantity["speed"], "3"]
SpeedBatchVec3: TypeAlias = Shaped[SpeedVec3, "*batch"]
SpeedBatchableVec3: TypeAlias = Shaped[SpeedVec3, "*#batch"]

SpecificEnergyScalar = Float[Quantity["specific_energy"], ""]
SpecificEnergyBatchScalar = Float[Quantity["specific_energy"], "*batch"]

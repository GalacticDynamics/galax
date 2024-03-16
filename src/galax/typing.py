"""Type hints for galax."""

# TODO: Finalize variable names and make everything public.
__all__: list[str] = []

from typing import TypeAlias

import astropy.units as u
from jaxtyping import Array, Float, Integer, Shaped

from unxt import Quantity

# =============================================================================

Unit: TypeAlias = u.Unit | u.UnitBase | u.CompositeUnit

# =============================================================================
# Scalars

AnyScalar = Shaped[Array, ""]
"""Any scalar."""

# An integer scalar.
IntScalar = Integer[Array, ""]
IntQScalar = Integer[Quantity, ""]

IntLike = IntScalar | int
"""An integer or integer scalar."""

# A float scalar.
FloatScalar = Float[Array, ""]
FloatQScalar = Float[Quantity, ""]

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
QVec1 = Float[Quantity, "1"]

# A 3-vector, e.g. q=(x, y, z) or p=(vx, vy, vz).
Vec3 = Float[Array, "3"]
QVec3 = Float[Quantity, "3"]

# A 3x3 matrix
Matrix33 = Float[Array, "3 3"]
QMatrix33 = Float[Quantity, "3 3"]

Vec4 = Float[Array, "4"]
"""A 4-vector e.g. w=(t, x, y, z)."""

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
QVecTime = Float[Quantity, "time"]
VecTime3 = Float[Vec3, "time"]
VecTime6 = Float[Vec6, "time"]
VecTime7 = Float[Vec7, "time"]

# -----------------------------------------------------------------------------
# Vector Batches

# -----------------
# Scalars

BatchIntScalar = Shaped[IntScalar, "*batch"]

BatchableIntLike = BatchIntScalar | IntLike

BroadBatchFloatScalar = Shaped[FloatScalar, "*#batch"]
BroadBatchFloatQScalar = Shaped[FloatQScalar, "*#batch"]

BatchFloatScalar = Shaped[FloatScalar, "*batch"]
BatchFloatQScalar = Shaped[FloatQScalar, "*batch"]

BatchableFloatLike = BatchFloatScalar | FloatLike

BatchRealScalar = Shaped[RealScalar, "*batch"]
BatchableRealScalar = Shaped[RealScalar, "*#batch"]
BatchRealQScalar = Shaped[RealQScalar, "*batch"]
BatchableRealQScalar = Shaped[RealQScalar, "*#batch"]

BatchableRealScalarLike = BatchableRealScalar | RealScalarLike

# -----------------
# Batched

BroadBatchVec1 = Shaped[Vec1, "*#batch"]
BatchVec1 = Shaped[Vec1, "*batch"]
"""Zero or more batches of 1-vectors."""

# Zero or more batches of 3-vectors.
BroadBatchVec3 = Shaped[Vec3, "*#batch"]
BroadBatchQVec3 = Shaped[QVec3, "*#batch"]
BatchVec3 = Shaped[Vec3, "*batch"]
BatchQVec3 = Shaped[QVec3, "*batch"]

# Zero or more batches of 3x3 matrices.
BatchMatrix33 = Shaped[Matrix33, "*batch"]
BatchQMatrix33 = Shaped[QMatrix33, "*batch"]

BroadBatchVec6 = Shaped[Vec6, "*#batch"]
BatchVec6 = Shaped[Vec6, "*batch"]
"""Zero or more batches of 6-vectors."""

BroadBatchVec7 = Shaped[Vec7, "*#batch"]
BatchVec7 = Shaped[Vec7, "*batch"]
"""Zero or more batches of 7-vectors."""

# -----------------
# Specific

# Zero or more batches of time vectors.
BroadBatchVecTime = Shaped[VecTime, "*#batch"]
BatchVecTime = Shaped[VecTime, "*batch"]
BatchVecTime6 = Shaped[VecTime6, "*batch"]
BatchVecTime7 = Shaped[VecTime7, "*batch"]
BroadBatchQVecTime = Shaped[QVecTime, "*#batch"]
BatchQVecTime = Shaped[QVecTime, "*batch"]

BroadBatchVecTime3 = Shaped[VecTime3, "*#batch"]

# -----------------
# Any Shape

# A float array with any shape.
FloatArrayAnyShape = Float[Array, "..."]
FloatQAnyShape = Float[Quantity, "..."]

IntArrayAnyShape = Integer[Array, "..."]
"""An integer array with any shape."""

ArrayAnyShape = Shaped[Array, "..."]

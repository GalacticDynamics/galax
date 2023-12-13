"""Type hints for galax."""

# TODO: Finalize variable names and make everything public.
__all__: list[str] = []

from typing import TypeAlias

import astropy.units as u
from jaxtyping import Array, Float, Integer, Shaped

# =============================================================================

Unit: TypeAlias = u.Unit | u.UnitBase | u.CompositeUnit

# =============================================================================
# Scalars

AnyScalar = Shaped[Array, ""]
"""Any scalar."""

IntScalar = Integer[Array, ""]
"""An integer scalar."""

IntLike = IntScalar | int
"""An integer or integer scalar."""

FloatScalar = Float[Array, ""]
"""A float scalar."""

FloatLike = FloatScalar | float | int
"""A float(/int) or float scalar."""

FloatOrIntScalar = FloatScalar | IntScalar
"""A float or integer scalar."""

FloatOrIntScalarLike = FloatLike | IntLike
"""A float or integer or float(/int) scalar."""


# =============================================================================
# Vectors

# -----------------------------------------------------------------------------
# Shaped

Vec3 = Float[Array, "3"]
"""A 3-vector, e.g. q=(x, y, z) or p=(vx, vy, vz)."""

Matrix33 = Float[Array, "3 3"]
"""A 3x3 matrix."""

Vec6 = Float[Array, "6"]
"""A 6-vector e.g. qp=(x, y, z, vx, vy, vz)."""

Vec7 = Float[Array, "7"]
"""A 7-vector e.g. w=(x, y, z, vx, vy, vz, t)."""

# -----------------------------------------------------------------------------
# Vector Batches

# -----------------
# Scalars

BatchIntScalar = Shaped[IntScalar, "*batch"]

BatchableIntLike = BatchIntScalar | IntLike

BatchFloatScalar = Shaped[FloatScalar, "*batch"]

BatchableFloatLike = BatchFloatScalar | FloatLike

BatchFloatOrIntScalar = Shaped[FloatOrIntScalar, "*batch"]

BatchableFloatOrIntScalarLike = BatchFloatOrIntScalar | FloatOrIntScalarLike

# -----------------
# Batched

BatchVec3 = Shaped[Vec3, "*batch"]
"""Zero or more batches of 3-vectors."""

BatchMatrix33 = Shaped[Matrix33, "*batch"]
"""Zero or more batches of 3x3 matrices."""

BatchVec6 = Shaped[Vec6, "*batch"]
"""Zero or more batches of 6-vectors."""

BatchVec7 = Shaped[Vec7, "*batch"]
"""Zero or more batches of 7-vectors."""

# -----------------
# Any Shape

FloatArrayAnyShape = Float[Array, "..."]
"""A float array with any shape."""

IntArrayAnyShape = Integer[Array, "..."]
"""An integer array with any shape."""

ArrayAnyShape = Shaped[Array, "..."]

# =============================================================================
# Specific Vectors

VecN = Float[Array, "N"]
"""An (N,)-vector."""

TimeVector = Float[Array, "time"]
"""A vector of times."""

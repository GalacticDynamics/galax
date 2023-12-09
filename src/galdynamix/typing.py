"""Type hints for galdynamix."""

# TODO: Finalize variable names and make everything public.
__all__: list[str] = []

from jaxtyping import Array, Float, Integer, Scalar, Shaped

# =============================================================================
# Scalars

IntScalar = Integer[Scalar, ""]
"""An integer scalar."""

IntLike = IntScalar | int
"""An integer or integer scalar."""

FloatScalar = Float[Scalar, ""]
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

BatchFloatScalar = Shaped[FloatScalar, "*batch"]

BatchableFloatLike = BatchFloatScalar | FloatLike

# -----------------
# Shaped

BatchVec3 = Shaped[Vec3, "*batch"]
"""Zero or more batches of 3-vectors."""

BatchVec6 = Shaped[Vec6, "*batch"]
"""Zero or more batches of 6-vectors."""

BatchVec7 = Shaped[Vec7, "*batch"]
"""Zero or more batches of 7-vectors."""

ArrayAnyShape = Float[Array, "..."]
"""An array with any shape."""

# =============================================================================
# Specific Vectors

VecN = Float[Array, "N"]
"""An (N,)-vector."""

TimeVector = Float[Array, "time"]
"""A vector of times."""

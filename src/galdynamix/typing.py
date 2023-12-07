"""Type hints for galdynamix."""

# TODO: Finalize variable names and make everything public.
__all__: list[str] = []

from jaxtyping import Array, Float, Integer, Scalar, Shaped

# =============================================================================
# Scalars

IntegerScalar = Integer[Scalar, ""]
"""An integer scalar."""

IntegerLike = IntegerScalar | int

FloatScalar = Float[Scalar, ""]
"""A float scalar."""

FloatLike = FloatScalar | float

# =============================================================================
# Vectors

# -----------------------------------------------------------------------------

Vec3 = Float[Array, "3"]
"""A 3-vector, e.g. q=(x, y, z) or p=(vx, vy, vz)."""

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

BatchVec3 = Shaped[Vec3, "*batch"]
"""Zero or more batches of 3-vectors."""

BatchVec6 = Shaped[Vec6, "*batch"]
"""Zero or more batches of 6-vectors."""

BatchVec7 = Shaped[Vec7, "*batch"]
"""Zero or more batches of 7-vectors."""

VecN = Float[Array, "N"]

ArrayAnyShape = Float[Array, "..."]

# =============================================================================
# Specific Vectors

TimeVector = Float[Array, "time"]

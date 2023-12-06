"""Type hints for galdynamix."""

__all__: list[str] = ["FloatScalar"]

from jaxtyping import Array, Float, Integer, Scalar

# =============================================================================
# Scalars

IntegerScalar = Integer[Scalar, ""]
"""An integer scalar."""

IntegerLike = IntegerScalar | int

FloatScalar = Float[Scalar, ""]
"""A float scalar."""

# =============================================================================
# Vectors

Vector3 = Float[Array, "3"]
"""A 3-vector, e.g. q=(x, y, z) or p=(vx, vy, vz)."""

Vector6 = Float[Array, "6"]
"""A 6-vector e.g. qp=(x, y, z, vx, vy, vz)."""

Vector7 = Float[Array, "7"]
"""A 7-vector e.g. w=(x, y, z, vx, vy, vz, t)."""

VectorN3 = Float[Vector3, "N"]
"""A batch of 3-vectors."""

VectorN6 = Float[Vector6, "N"]
"""A batch of 6-vectors."""

VectorN7 = Float[Vector7, "N"]
"""A batch of 7-vectors."""

VectorN = Float[Array, "N"]

ArrayAnyShape = Float[Array, "..."]

# =============================================================================
# Specific Vectors

TimeVector = Float[Array, "time"]

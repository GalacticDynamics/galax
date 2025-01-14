"""Type hints for galax.

As indicated by `__all__`, this module does not export any names. The type hints
defined here may be changed or removed without notice. They are intended for use
in other modules within the `galax` package.

Notes
-----
- "Bt" stands for "batch", which in `jaxtyping` is '#batch'.
- "BBt" stands for broadcast batchable, which in `jaxtyping` is '*#batch'.

"""

__all__: list[str] = []

from typing import TypeAlias

import astropy.units as apyu
from jaxtyping import Array, Float, Integer, Scalar, Shaped

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

BtScalar: TypeAlias = Shaped[Scalar, "*batch"]
BBtScalar: TypeAlias = Shaped[Scalar, "*#batch"]

BtFloatScalar: TypeAlias = Shaped[FloatScalar, "*batch"]
BBtFloatScalar: TypeAlias = Shaped[FloatScalar, "*#batch"]

BBtFloatQScalar: TypeAlias = Shaped[FloatQScalar, "*#batch"]

BtFloatQScalar: TypeAlias = Shaped[FloatQScalar, "*batch"]

BtRealQScalar: TypeAlias = Shaped[RealQScalar, "*batch"]
BBtRealScalar: TypeAlias = Shaped[RealScalar, "*#batch"]
BBtRealQScalar: TypeAlias = Shaped[RealQScalar, "*#batch"]

# -----------------
# Batched

# Zero or more batches of 3-vectors.
BtVec3: TypeAlias = Shaped[Vec3, "*batch"]
BtQVec3: TypeAlias = Shaped[QVec3, "*batch"]

# Zero or more batches of 6-vectors
BtVec6: TypeAlias = Shaped[Vec6, "*batch"]
BBtVec6: TypeAlias = Shaped[Vec6, "*#batch"]

# Zero or more batches of 7-vectors
BtVec7: TypeAlias = Shaped[Vec7, "*batch"]
BBtVec7: TypeAlias = Shaped[Vec7, "*#batch"]

# -----------------
# Any Shape

# A float array with any shape.
FloatQAnyShape: TypeAlias = Float[AbstractQuantity, "..."]


# =============================================================================
# Matrices

QMatrix33: TypeAlias = Float[AbstractQuantity, "3 3"]
BtQMatrix33: TypeAlias = Shaped[QMatrix33, "*batch"]


# =============================================================================

MassScalar: TypeAlias = Shaped[u.Quantity["mass"], ""]
MassBtScalar: TypeAlias = Shaped[u.Quantity["mass"], "*batch"]
MassBBtScalar: TypeAlias = Shaped[u.Quantity["mass"], "*#batch"]

TimeScalar: TypeAlias = Shaped[u.Quantity["time"], ""]
TimeBtScalar: TypeAlias = Shaped[u.Quantity["time"], "*batch"]
TimeBBtScalar: TypeAlias = Shaped[u.Quantity["time"], "*#batch"]

LengthVec3: TypeAlias = Shaped[u.Quantity["length"], "3"]
LengthBtVec3: TypeAlias = Shaped[LengthVec3, "*batch"]
LengthBBtVec3: TypeAlias = Shaped[LengthVec3, "*#batch"]

SpeedVec3: TypeAlias = Shaped[u.Quantity["speed"], "3"]
SpeedBtVec3: TypeAlias = Shaped[SpeedVec3, "*batch"]
SpeedBBtVec3: TypeAlias = Shaped[SpeedVec3, "*#batch"]

SpecificEnergyScalar: TypeAlias = Float[u.Quantity["specific_energy"], ""]
SpecificEnergyBtScalar: TypeAlias = Float[u.Quantity["specific_energy"], "*batch"]


# =============================================================================

Qarr: TypeAlias = Shaped[Array, "3"]
BtQarr: TypeAlias = Shaped[Qarr, "*batch"]
BBtQarr: TypeAlias = Shaped[Qarr, "*#batch"]

Q: TypeAlias = Shaped[AbstractQuantity, "3"]
BtQ: TypeAlias = Shaped[Q, "*batch"]
BBtQ: TypeAlias = Shaped[Q, "*#batch"]

Parr: TypeAlias = Shaped[Array, "3"]
BtParr: TypeAlias = Shaped[Parr, "*batch"]
BBtParr: TypeAlias = Shaped[Parr, "*#batch"]

P: TypeAlias = Shaped[AbstractQuantity, "3"]
BtP: TypeAlias = Shaped[P, "*batch"]
BBtP: TypeAlias = Shaped[P, "*#batch"]

Aarr: TypeAlias = Shaped[Array, "3"]
BtAarr: TypeAlias = Shaped[Aarr, "*batch"]

QParr: TypeAlias = tuple[Qarr, Parr]
BtQParr: TypeAlias = tuple[BtQarr, BtParr]
BBtQParr: TypeAlias = tuple[BBtQarr, BBtParr]

QP: TypeAlias = tuple[Q, P]
BtQP: TypeAlias = tuple[BtQ, BtP]
BBtQP: TypeAlias = tuple[BBtQ, BBtP]

PAarr: TypeAlias = tuple[Parr, Aarr]
BtPAarr: TypeAlias = tuple[BtParr, BtAarr]

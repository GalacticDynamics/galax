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
IntSz0: TypeAlias = Integer[Array, ""]
IntQSz0: TypeAlias = Integer[AbstractQuantity, ""]
IntLike: TypeAlias = IntSz0 | int

# A float scalar.
FloatSz0: TypeAlias = Float[Array, ""]
FloatQSz0: TypeAlias = Float[AbstractQuantity, ""]

FloatLike: TypeAlias = FloatSz0 | float | int
"""A float(/int) or float scalar."""

# A float or integer scalar.
RealSz0: TypeAlias = FloatSz0 | IntSz0
RealQSz0: TypeAlias = FloatQSz0 | IntQSz0

# A float or integer or float(/int) scalar.
RealSz0Like: TypeAlias = FloatLike | IntLike


# =============================================================================
# Vectors

# -----------------------------------------------------------------------------
# Shaped

# 1-vector
Sz1: TypeAlias = Float[Array, "1"]
QSz1: TypeAlias = Float[AbstractQuantity, "1"]

# A 3-vector, e.g. q=(x, y, z) or p=(vx, vy, vz).
Sz3: TypeAlias = Float[Array, "3"]
QSz3: TypeAlias = Float[AbstractQuantity, "3"]

Sz6: TypeAlias = Float[Array, "6"]
"""A 6-vector e.g. w=(x, y, z, vx, vy, vz)."""

Sz7: TypeAlias = Float[Array, "7"]
"""A 7-vector e.g. w=(x, y, z, vx, vy, vz, t)."""

SzN: TypeAlias = Float[Array, "N"]
"""An (N,)-vector."""

# -----------------
# Specific

# Time vector
SzTime: TypeAlias = Float[Array, "time"]
QSzTime: TypeAlias = Float[AbstractQuantity, "time"]

# -----------------------------------------------------------------------------
# Vector Batches

# -----------------
# Scalars

BtSz0: TypeAlias = Shaped[Scalar, "*batch"]
BBtSz0: TypeAlias = Shaped[Scalar, "*#batch"]

BtFloatSz0: TypeAlias = Shaped[FloatSz0, "*batch"]
BBtFloatSz0: TypeAlias = Shaped[FloatSz0, "*#batch"]

BBtFloatQSz0: TypeAlias = Shaped[FloatQSz0, "*#batch"]

BtFloatQSz0: TypeAlias = Shaped[FloatQSz0, "*batch"]

BtRealQSz0: TypeAlias = Shaped[RealQSz0, "*batch"]
BBtRealSz0: TypeAlias = Shaped[RealSz0, "*#batch"]
BBtRealQSz0: TypeAlias = Shaped[RealQSz0, "*#batch"]

# -----------------
# Batched

# Zero or more batches of 3-vectors.
BtSz3: TypeAlias = Shaped[Sz3, "*batch"]
BtQSz3: TypeAlias = Shaped[QSz3, "*batch"]

# Zero or more batches of 6-vectors
BtSz6: TypeAlias = Shaped[Sz6, "*batch"]
BBtSz6: TypeAlias = Shaped[Sz6, "*#batch"]

# Zero or more batches of 7-vectors
BtSz7: TypeAlias = Shaped[Sz7, "*batch"]
BBtSz7: TypeAlias = Shaped[Sz7, "*#batch"]

# -----------------
# Any Shape

# A float array with any shape.
FloatQSzAny: TypeAlias = Float[AbstractQuantity, "..."]


# =============================================================================
# Matrices

QSz33: TypeAlias = Float[AbstractQuantity, "3 3"]
BtQSz33: TypeAlias = Shaped[QSz33, "*batch"]


# =============================================================================

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

"""Base classes for operators on coordinates and potentials."""

__all__ = ["simplify_op"]

from functools import singledispatch
from typing import cast

from plum import dispatch

from coordinax import Abstract3DVector, Cartesian3DVector

from .base import AbstractOperator
from galax.typing import QVec3, RealScalar


@singledispatch
def simplify_op(op: AbstractOperator, /) -> AbstractOperator:
    """Simplify an operator."""
    return op


@dispatch
def frame_transformation_into(
    op: AbstractOperator, x: Abstract3DVector, t: RealScalar, /
) -> tuple[Abstract3DVector, RealScalar]:
    """Apply the operator to the coordinates."""
    return op(x, t)


@dispatch
def frame_transformation_into(
    op: AbstractOperator, x: QVec3, t: RealScalar, /
) -> tuple[Cartesian3DVector, RealScalar]:
    """Apply the operator to the coordinates."""
    cart = Cartesian3DVector.constructor(x)
    return op(cart, t)


@dispatch  # type: ignore[misc]
def frame_transformation_outof(
    op: AbstractOperator, x: Abstract3DVector, t: RealScalar, /
) -> tuple[Abstract3DVector, RealScalar]:
    """Undo."""
    return cast("tuple[Abstract3DVector, RealScalar]", op.inverse(x, t))

"""Base classes for operators on coordinates and potentials."""

__all__ = ["AbstractOperator", "IdentityOperator"]

from typing import Literal, final

from jax_quantity import Quantity
from vector import AbstractVector

from .base import AbstractOperator, op_call_dispatch


@final
class IdentityOperator(AbstractOperator):
    """Identity operation."""

    @op_call_dispatch(precedence=1)  # type: ignore[misc]
    def __call__(
        self: "IdentityOperator", x: AbstractVector, t: Quantity["time"], /
    ) -> tuple[AbstractVector, Quantity["time"]]:
        """Apply the Identity operation.

        This is the identity operation, which does nothing to the input.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector
        >>> from galax.coordinates.operators import IdentityOperator

        >>> q = Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> op = IdentityOperator()
        >>> op(q, t)
        (Cartesian3DVector( ... ), Quantity['time'](Array(0, ...), unit='Gyr'))
        """
        return x, t

    @property
    def is_inertial(self) -> Literal[True]:
        """Identity operation is an inertial-frame preserving transform.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector
        >>> from galax.coordinates.operators import IdentityOperator

        >>> q = Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> op = IdentityOperator()
        >>> op.is_inertial
        True
        """
        return True

    @property
    def inverse(self) -> "IdentityOperator":
        """The inverse of the operator.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector
        >>> from galax.coordinates.operators import IdentityOperator

        >>> q = Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> op = IdentityOperator()
        >>> op.inverse
        IdentityOperator()
        """
        return self

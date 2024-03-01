"""Base classes for operators on coordinates and potentials."""

__all__ = ["AbstractCompositeOperator"]

from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

from jax_quantity import Quantity
from vector import AbstractVector

from .base import AbstractOperator, op_call_dispatch
from galax.utils.dataclasses import _DataclassInstance

if TYPE_CHECKING:
    from typing import Self


@runtime_checkable
class HasOperatorsAttr(_DataclassInstance, Protocol):
    operators: tuple[AbstractOperator, ...]


class AbstractCompositeOperator(AbstractOperator):
    """Abstract Composite Operator.

    This is the base class for all composite operations.

    See Also
    --------
    :class:`galax.coordinates.frame.OperatorSequence`
    :class:`galax.coordinates.frame.GalileanOperator`
    """

    # ===========================================
    # Operator

    # TODO: how to have the `operators` attribute in a way that allows for both
    # writeable (in the constructor) and read-only (as a property) subclasses.

    @op_call_dispatch(precedence=1)  # type: ignore[misc]
    def __call__(
        self: "AbstractCompositeOperator", x: AbstractVector, t: Quantity["time"], /
    ) -> tuple[AbstractVector, Quantity["time"]]:
        """Apply the operator to the coordinates."""
        for op in self.operators:
            x, t = op(x, t)
        return x, t

    @property
    def is_inertial(self: HasOperatorsAttr) -> bool:
        """Whether the operations maintain an inertial reference frame."""
        return all(op.is_inertial for op in self.operators)

    @property
    def inverse(self: HasOperatorsAttr) -> "AbstractCompositeOperator":
        """The inverse of the operator."""
        from .sequential import OperatorSequence

        return OperatorSequence(tuple(op.inverse for op in reversed(self.operators)))

    # ===========================================
    # Sequence

    @overload
    def __getitem__(self, key: int) -> AbstractOperator:
        ...

    @overload
    def __getitem__(self, key: slice) -> "Self":
        ...

    def __getitem__(self, key: int | slice) -> "AbstractOperator | Self":
        ops = self.operators[key]
        if isinstance(ops, AbstractOperator):
            return ops
        return replace(self, operators=ops)

    def __iter__(self: HasOperatorsAttr) -> Iterator[AbstractOperator]:
        return iter(self.operators)

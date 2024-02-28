"""Sequence of Operators."""

__all__ = ["OperatorSequence"]

from dataclasses import replace
from typing import Any, final

import equinox as eqx

from .base import AbstractOperator
from .composite import AbstractCompositeOperator
from .funcs import simplify_op
from .identity import IdentityOperator


def _converter_seq(inp: Any) -> tuple[AbstractOperator, ...]:
    if isinstance(inp, tuple):
        return inp
    if isinstance(inp, OperatorSequence):
        return inp.operators
    if isinstance(inp, AbstractOperator):
        return (inp,)

    raise TypeError


@final
class OperatorSequence(AbstractCompositeOperator):
    """Sequence of operations.

    This is the composite operator that represents a sequence of operations to
    be applied in order.

    Parameters
    ----------
    operators : tuple[AbstractOperator, ...]
        The sequence of operators to apply.

    Examples
    --------
    >>> from jax_quantity import Quantity
    >>> import galax.coordinates.operators as gco

    >>> shift = gco.GalileanSpatialTranslationOperator(Quantity([1, 2, 3], "kpc"))
    >>> boost = gco.GalileanBoostOperator(Quantity([10, 20, 30], "km/s"))
    >>> seq = gco.OperatorSequence((shift, boost))
    >>> seq
    OperatorSequence(
      operators=( GalileanSpatialTranslationOperator( ... ),
                  GalileanBoostOperator( ... ) )
    )

    A sequence of operators can also be constructed by ``|``:

    >>> seq2 = shift | boost
    >>> seq2
    OperatorSequence(
      operators=( GalileanSpatialTranslationOperator( ... ),
                  GalileanBoostOperator( ... ) )
    )

    The sequence of operators can be simplified. For this example, we
    add an identity operator to the sequence:

    >>> seq3 = seq2 | gco.IdentityOperator()
    >>> seq3
    OperatorSequence(
      operators=( GalileanSpatialTranslationOperator( ... ),
                  GalileanBoostOperator( ... ),
                  IdentityOperator() )
    )

    >>> gco.simplify_op(seq3)
    OperatorSequence(
      operators=( GalileanSpatialTranslationOperator( ... ),
                  GalileanBoostOperator( ... ) )
    )
    """

    operators: tuple[AbstractOperator, ...] = eqx.field(converter=_converter_seq)

    def __or__(self, other: AbstractOperator) -> "OperatorSequence":
        """Compose with another operator."""
        if isinstance(other, type(self)):
            return replace(self, operators=(*self, *other.operators))
        return replace(self, operators=(*self, other))

    def __ror__(self, other: AbstractOperator) -> "OperatorSequence":
        return replace(self, operators=(other, *self))


#####################################################################
# Functions


@simplify_op.register
def _simplify_op(seq: OperatorSequence, /) -> OperatorSequence:
    """Simplify a sequence of Operators.

    This simplifies the sequence of operators by removing any that reduce to
    the Identity operator.
    """
    # Iterate through the operators, simplifying that operator, then filtering
    # out any that reduce to the Identity.
    # TODO: this doesn't do any type of operator fusion, e.g. a
    # ``GalileanRotationOperator | GalileanTranslationOperator |
    # GalileanBoostOperator => GalileanOperator``
    return OperatorSequence(
        tuple(
            sop
            for op in seq.operators
            if not isinstance((sop := simplify_op(op)), IdentityOperator)
        )
    )

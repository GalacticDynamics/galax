"""Base classes for operators on coordinates and potentials."""

__all__ = ["AbstractOperator"]

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
from plum import dispatch

from jax_quantity import Quantity
from vector import Abstract3DVector, AbstractVector, FourVector

from galax.coordinates._psp.base import AbstractPhaseSpacePositionBase
from galax.coordinates._psp.psp import PhaseSpacePosition
from galax.coordinates._psp.pspt import AbstractPhaseSpaceTimePosition
from galax.utils.dataclasses import dataclass_items

if TYPE_CHECKING:
    from .sequential import OperatorSequence

T = TypeVar("T")


class AbstractOperator(eqx.Module):  # type: ignore[misc]
    """Abstract base class for operators on coordinates and potentials."""

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch  # type: ignore[misc]
    def constructor(
        cls: "type[AbstractOperator]", obj: Mapping[str, Any], /
    ) -> "AbstractOperator":
        """Construct from a mapping.

        Parameters
        ----------
        obj : Mapping[str, Any]
            The object to construct from.

        Returns
        -------
        AbstractOperator
            The constructed operator.

        Examples
        --------
        >>> import galax.coordinates.operators as gco
        >>> operators = gco.IdentityOperator() | gco.IdentityOperator()
        >>> gco.OperatorSequence.constructor({"operators": operators})
        OperatorSequence(operators=(IdentityOperator(), IdentityOperator()))
        """
        return cls(**obj)

    # -------------------------------------------

    @dispatch
    def __call__(
        self: "AbstractOperator",
        x: AbstractVector,  # noqa: ARG002
        t: Quantity["time"],  # noqa: ARG002
        /,
    ) -> tuple[AbstractVector, Quantity["time"]]:
        """Apply the operator to the coordinates."""
        msg = "Operators apply to 3+ dimensional vectors"
        raise TypeError(msg)

    @dispatch
    def __call__(
        self: "AbstractOperator",
        q: Abstract3DVector,  # noqa: ARG002
        t: Quantity["time"],  # noqa: ARG002
        /,
    ) -> tuple[Abstract3DVector, Quantity["time"]]:
        """Apply the operator to the coordinates."""
        msg = "implement this method in the subclass"
        raise NotImplementedError(msg)

    @dispatch
    def __call__(self: "AbstractOperator", x: FourVector, /) -> FourVector:
        """Apply the operator to the coordinates."""
        q, t = self(x.q, x.t)  # redispatch on (q, t)
        return FourVector(q=q, t=t)

    @dispatch
    def __call__(
        self: "AbstractOperator",
        x: AbstractPhaseSpacePositionBase,  # noqa: ARG002
        t: Quantity["time"],  # noqa: ARG002
        /,
    ) -> tuple[AbstractPhaseSpacePositionBase, Quantity["time"]]:
        """Apply the operator to a phase-space position."""
        msg = "implement this method in the subclass"
        raise NotImplementedError(msg)

    @dispatch
    def __call__(
        self: "AbstractOperator", x: AbstractPhaseSpaceTimePosition, /
    ) -> AbstractPhaseSpaceTimePosition:
        """Apply the operator to a phase-space-time position.

        This method calls the method that operates on
        ``AbstractPhaseSpacePositionBase`` by separating the time component from
        the rest of the phase-space position.  Subclasses can implement that
        method to avoid having to implement for both phase-space-time and
        phase-space positions.  Alternatively, they can implement this method
        directly to avoid redispatching.
        """
        # redispatch on (psp, t)
        psp, t = self(PhaseSpacePosition(q=x.q, p=x.p), x.t)
        return replace(x, q=psp.q, p=psp.p, t=t)

    # -------------------------------------------

    @property
    @abstractmethod
    def is_inertial(self) -> bool:
        """Whether the operation maintains an inertial reference frame."""
        ...

    @property
    @abstractmethod
    def inverse(self) -> "AbstractOperator":
        """The inverse of the operator."""
        ...

    # ===========================================
    # Sequence

    def __or__(self, other: "AbstractOperator") -> "OperatorSequence":
        """Compose with another operator."""
        from .sequential import OperatorSequence

        if isinstance(other, OperatorSequence):
            return other.__ror__(self)
        return OperatorSequence((self, other))


op_call_dispatch = AbstractOperator.__call__.dispatch  # type: ignore[attr-defined]


# TODO: move to the class in py3.11+
@AbstractOperator.constructor._f.register  # type: ignore[misc]  # noqa: SLF001
def constructor(
    cls: type[AbstractOperator], obj: AbstractOperator, /
) -> AbstractOperator:
    """Construct an operator from another operator.

    Parameters
    ----------
    obj : :class:`galax.coordinates.operators.AbstractOperator`
        The object to construct from.
    """
    if not isinstance(obj, cls):
        msg = f"Cannot construct {cls} from {type(obj)}."
        raise TypeError(msg)

    # avoid copying if the types are the same. Isinstance is not strict
    # enough, so we use type() instead.
    if type(obj) is cls:  # pylint: disable=unidiomatic-typecheck
        return obj

    return cls(**dict(dataclass_items(obj)))

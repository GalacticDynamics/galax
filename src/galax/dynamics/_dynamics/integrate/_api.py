__all__ = ["Integrator"]

from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from unxt import AbstractUnitSystem

import galax.coordinates as gc
import galax.typing as gt
from galax.utils.dataclasses import _DataclassInstance

SaveT: TypeAlias = gt.BatchQVecTime | gt.QVecTime | gt.BatchVecTime | gt.VecTime


@runtime_checkable
class FCallable(Protocol):
    """Protocol for the integration callable."""

    def __call__(self, t: gt.FloatScalar, w: gt.Vec6, args: tuple[Any, ...]) -> gt.Vec6:
        """Integration function.

        Parameters
        ----------
        t : float
            The time. This is the integration variable.
        w : Array[float, (6,)]
            The position and velocity.
        args : tuple[Any, ...]
            Additional arguments.

        Returns
        -------
        Array[float, (6,)]
            Velocity and acceleration [v (3,), a (3,)].
        """
        ...


@runtime_checkable
class Integrator(_DataclassInstance, Protocol):
    """:class:`typing.Protocol` for integrators.

    The integrators are classes that are used to integrate the equations of
    motion.

    .. note::

        Integrators should NOT be stateful (i.e., they must not have attributes
        that change).
    """

    # TODO: shape hint of the return type
    def __call__(
        self,
        F: FCallable,
        w0: gc.AbstractPhaseSpacePosition | gt.BatchVec6,
        t0: gt.FloatQScalar | gt.FloatScalar,
        t1: gt.FloatQScalar | gt.FloatScalar,
        /,
        savet: SaveT | None = None,
        *,
        units: AbstractUnitSystem,
        interpolated: Literal[False, True] = False,
    ) -> gc.PhaseSpacePosition | gc.InterpolatedPhaseSpacePosition:
        """Integrate.

        Parameters
        ----------
        F : FCallable, positional-only
            The function to integrate.
            (t, w, args) -> (v, a).
        w0 : AbstractPhaseSpacePosition | Array[float, (6,)], positional-only
            Initial conditions ``[q, p]``.
        t0, t1 : Quantity, positional-only
            Initial and final times.

        savet : (Quantity | Array)[float, (T,)] | None, optional
            Times to return the computation.
            If `None`, the solution is returned at the final time.

        units : `unxt.AbstractUnitSystem`, keyword-only
            The unit system to use.

        Returns
        -------
        PhaseSpacePosition[float, (T, 7)]
            The solution of the integrator [q, p, t], where q, p are the
            generalized 3-coordinates.
        """
        ...

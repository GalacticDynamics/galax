__all__ = ["Integrator"]

from typing import Any, Protocol, runtime_checkable

from galax.typing import FloatScalar, Vec6, VecTime, VecTime7
from galax.utils.dataclasses import _DataclassInstance


@runtime_checkable
class FCallable(Protocol):
    """Protocol for the integration callable."""

    def __call__(self, t: FloatScalar, w: Vec6, args: tuple[Any, ...]) -> Vec6:
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
    They must not be stateful since they are used in a functional way.
    """

    def __call__(self, F: FCallable, w0: Vec6, /, ts: VecTime) -> VecTime7:
        """Integrate.

        Parameters
        ----------
        F : FCallable, positional-only
            The function to integrate.
            (t, w, args) -> (v, a).
        w0 : Array[float, (6,)], positional-only
            Initial conditions ``[q, p]``.

        ts : Array[float, (T,)]
            Times to return the computation.
            It's necessary to at least provide the initial and final times.

        Returns
        -------
        Array[float, (T, 7)]
            The solution of the integrator [q, p, t], where q, p are the
            generalized 3-coordinates.
        """
        ...

__all__ = ["Integrator", "AbstractIntegrator"]

import abc
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
from jaxtyping import Array, Float

from galax.typing import FloatScalar, Vec6
from galax.utils.dataclasses import _DataclassInstance


@runtime_checkable
class FCallable(Protocol):
    def __call__(self, t: FloatScalar, qp: Vec6, args: tuple[Any, ...]) -> Vec6:
        """Integration function.

        Parameters
        ----------
        t : float
            The time.
        qp : Array[float, (6,)]
            The position and velocity.
        args : tuple
            Additional arguments.

        Returns
        -------
        Array[float, (6,)]
            [v (3,), a (3,)].
        """
        ...


@runtime_checkable
class Integrator(_DataclassInstance, Protocol):
    """:class:`typing.Protocol` for integrators.

    The integrators are classes that are used to integrate the equations of
    motion.
    They must not be stateful since they are used in a functional way.
    """

    def __call__(
        self, F: FCallable, qp0: Vec6, /, ts: Float[Array, "T"] | None
    ) -> Float[Array, "R 7"]:
        """Integrate.

        Parameters
        ----------
        F : FCallable, positional-only
            The function to integrate.
            (t, qp, args) -> (v, a).
        qp0 : Array[float, (6,)], positional-only
            Initial conditions ``[q, p]``.

        ts : Array[float, (T,)] | None
            Times to return the computation.
            It's necessary to at least provide the initial and final times.

        Returns
        -------
        Array[float, (R, 7)]
            The solution of the integrator [q, p, t], where q, p are the
            generalized 3-coordinates.
        """
        ...


class AbstractIntegrator(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract base class for integrators.

    This class is the base for the hierarchy of concrete integrator classes
    provided in this package. It is not necessary, but it is recommended, to
    inherit from this class to implement an integrator. The Protocol
    :class:`Integrator` must be implemented.

    The integrators are classes that are used to integrate the equations of
    motion.  They must not be stateful since they are used in a functional way.
    """

    @abc.abstractmethod
    def __call__(
        self,
        F: FCallable,
        qp0: Vec6,
        /,
        ts: Float[Array, "T"],
    ) -> Float[Array, "T 7"]:
        """Run the integrator.

        Parameters
        ----------
        F : FCallable, positional-only
            The function to integrate.
        qp0 : Array[float, (6,)], positional-only
            Initial conditions ``[q, p]``.

        ts : Array[float, (T,)] | None
            Times to return the computation.
            It's necessary to at least provide the initial and final times.

        Returns
        -------
        Array[float, (R, 7)]
            The solution of the integrator [q, p, t], where q, p are the
            generalized 3-coordinates.
        """
        ...

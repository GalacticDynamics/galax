__all__ = ["AbstractIntegrator"]

import abc
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
from jaxtyping import Array, Float

from galax.typing import FloatScalar, Vec6


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


class AbstractIntegrator(eqx.Module):  # type: ignore[misc]
    """Integrator Class.

    The integrators are classes that are used to integrate the equations of
    motion.
    They must not be stateful since they are used in a functional way.
    """

    @abc.abstractmethod
    def run(
        self,
        F: FCallable,
        qp0: Vec6,
        t0: FloatScalar,
        t1: FloatScalar,
        ts: Float[Array, "T"] | None,
    ) -> Float[Array, "R 7"]:
        """Run the integrator.

        .. todo::

            Have a better time parser.

        Parameters
        ----------
        F : FCallable
            The function to integrate.
        qp0 : Array[float, (6,)]
            Initial conditions ``[q, p]``.
        t0 : float
            Initial time.
        t1 : float
            Final time.
        ts : Array[float, (T,)] | None
            Times for the computation.

        Returns
        -------
        Array[float, (R, 7)]
            The solution of the integrator [q, p, t], where q, p are the
            generalized 3-coordinates.
        """
        ...

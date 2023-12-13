__all__ = ["AbstractIntegrator"]

import abc
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
from jaxtyping import Array, Float

from galax.typing import FloatScalar, Vec6, Vec7


@runtime_checkable
class FCallable(Protocol):
    def __call__(self, t: FloatScalar, qp: Vec6, args: tuple[Any, ...]) -> Vec7:
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
        Array[float, (7,)]
            [qp, t].
        """
        ...


class AbstractIntegrator(eqx.Module):  # type: ignore[misc]
    """Integrator Class."""

    F: FCallable
    """The function to integrate."""
    # TODO: should this be moved to be the first argument of the run method?

    @abc.abstractmethod
    def run(
        self,
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

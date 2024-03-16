__all__ = ["AbstractIntegrator"]

import abc

import equinox as eqx

from unxt import UnitSystem

from ._api import FCallable
from galax.coordinates import AbstractPhaseSpacePosition, PhaseSpacePosition
from galax.typing import BatchQVecTime, BatchVec6, BatchVecTime, QVecTime, VecTime


class AbstractIntegrator(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract base class for integrators.

    This class is the base for the hierarchy of concrete integrator classes
    provided in this package. It is not necessary, but it is recommended, to
    inherit from this class to implement an integrator. The Protocol
    :class:`Integrator` must be implemented.

    The integrators are classes that are used to integrate the equations of
    motion.  They must not be stateful since they are used in a functional way.
    """

    # TODO: shape hint of the return type
    @abc.abstractmethod
    def __call__(
        self,
        F: FCallable,
        w0: AbstractPhaseSpacePosition | BatchVec6,
        /,
        ts: BatchQVecTime | BatchVecTime | QVecTime | VecTime,
        *,
        units: UnitSystem,
    ) -> PhaseSpacePosition:
        """Run the integrator.

        Parameters
        ----------
        F : FCallable, positional-only
            The function to integrate.
        w0 : AbstractPhaseSpacePosition | Array[float, (6,)], positional-only
            Initial conditions ``[q, p]``.
        ts : (Quantity | Array)[float, (T,)]
            Times to return the computation.
            It's necessary to at least provide the initial and final times.
        units : UnitSystem
            The unit system to use.

        Returns
        -------
        PhaseSpacePosition[float, (time, 7)]
            The solution of the integrator [q, p, t], where q, p are the
            generalized 3-coordinates.
        """
        ...

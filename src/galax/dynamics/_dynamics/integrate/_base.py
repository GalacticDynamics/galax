__all__ = ["AbstractIntegrator"]

import abc
from typing import Literal

import equinox as eqx

from unxt import AbstractUnitSystem

import galax.coordinates as gc
import galax.typing as gt
from ._api import FCallable


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
        w0: gc.AbstractPhaseSpacePosition | gt.BatchVec6,
        t0: gt.FloatQScalar | gt.FloatScalar,
        t1: gt.FloatQScalar | gt.FloatScalar,
        /,
        savet: (
            gt.BatchQVecTime | gt.BatchVecTime | gt.QVecTime | gt.VecTime | None
        ) = None,
        *,
        units: AbstractUnitSystem,
        interpolated: Literal[False, True] = False,
    ) -> gc.PhaseSpacePosition | gc.InterpolatedPhaseSpacePosition:
        """Run the integrator.

        Parameters
        ----------
        F : FCallable, positional-only
            The function to integrate.
        w0 : AbstractPhaseSpacePosition | Array[float, (6,)], positional-only
            Initial conditions ``[q, p]``.
        t0, t1 : Quantity, positional-only
            Initial and final times.

        savet : (Quantity | Array)[float, (T,)] | None, optional
            Times to return the computation.  If `None`, the computation is
            returned only at the final time.

        units : `unxt.AbstractUnitSystem`
            The unit system to use.
        interpolated : bool, keyword-only
            Whether to return an interpolated solution.

        Returns
        -------
        PhaseSpacePosition[float, (time, 7)]
            The solution of the integrator [q, p, t], where q, p are the
            generalized 3-coordinates.

        Examples
        --------
        For this example, we will use the
        :class:`~galax.integrate.DiffraxIntegrator`

        First some imports:

        >>> import quaxed.array_api as xp
        >>> from unxt import Quantity
        >>> import unxt.unitsystems as usx
        >>> import galax.coordinates as gc
        >>> import galax.dynamics as gd
        >>> import galax.potential as gp

        Then we define initial conditions:

        >>> w0 = gc.PhaseSpacePosition(q=Quantity([10., 0., 0.], "kpc"),
        ...                            p=Quantity([0., 200., 0.], "km/s"))

        (Note that the ``t`` attribute is not used.)

        Now we can integrate the phase-space position for 1 Gyr, getting the
        final position.  The integrator accepts any function for the equations
        of motion.  Here we will reproduce what happens with orbit integrations.

        >>> pot = gp.HernquistPotential(m_tot=Quantity(1e12, "Msun"),
        ...                             r_s=Quantity(5, "kpc"), units="galactic")

        >>> integrator = gd.integrate.DiffraxIntegrator()
        >>> t0, t1 = Quantity(0, "Gyr"), Quantity(1, "Gyr")
        >>> w = integrator(pot._integrator_F, w0, t0, t1, units=usx.galactic)
        >>> w
        PhaseSpacePosition(
            q=CartesianPosition3D( ... ),
            p=CartesianVelocity3D( ... ),
            t=Quantity[...](value=f64[], unit=Unit("Myr"))
        )
        >>> w.shape
        ()

        We can also request the orbit at specific times:

        >>> ts = Quantity(xp.linspace(0, 1, 10), "Myr")  # 10 steps
        >>> ws = integrator(pot._integrator_F, w0, t0, t1, savet=ts, units=usx.galactic)
        >>> ws
        PhaseSpacePosition(
            q=CartesianPosition3D( ... ),
            p=CartesianVelocity3D( ... ),
            t=Quantity[...](value=f64[10], unit=Unit("Myr"))
        )
        >>> ws.shape
        (10,)

        The integrator can also be used to integrate a batch of initial
        conditions at once, returning a batch of final conditions (or a batch
        of conditions at the requested times):

        >>> w0 = gc.PhaseSpacePosition(q=Quantity([[10., 0, 0], [10., 0, 0]], "kpc"),
        ...                            p=Quantity([[0, 200, 0], [0, 200, 0]], "km/s"))
        >>> ws = integrator(pot._integrator_F, w0, t0, t1, units=usx.galactic)
        >>> ws.shape
        (2,)

        """
        ...

__all__ = ["AbstractIntegrator"]

import abc
from typing import Any, Literal, TypeVar

import equinox as eqx
from plum import overload

import quaxed.array_api as xp
from unxt import AbstractUnitSystem, Quantity, to_units_value

import galax.coordinates as gc
import galax.typing as gt
from ._api import SaveT, VectorField

Interp = TypeVar("Interp")


class AbstractIntegrator(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract base class for integrators.

    This class is the base for the hierarchy of concrete integrator classes
    provided in this package. It is not necessary, but it is recommended, to
    inherit from this class to implement an integrator. The Protocol
    :class:`Integrator` must be implemented.

    The integrators are classes that are used to integrate the equations of
    motion.  They must not be stateful since they are used in a functional way.
    """

    InterpolantClass: eqx.AbstractClassVar[type[gc.PhaseSpacePositionInterpolant]]

    @abc.abstractmethod
    def _call_implementation(
        self,
        F: VectorField,
        w0: gt.BatchVec6,
        t0: gt.FloatScalar,
        t1: gt.FloatScalar,
        ts: gt.BatchVecTime,
        /,
        interpolated: Literal[False, True],
    ) -> tuple[gt.BatchVecTime7, Any | None]:  # TODO: type hint Interpolant
        """Integrator implementation."""
        ...

    def _process_interpolation(
        self, interp: Interp, w0: gt.BatchVec6
    ) -> tuple[Interp, int]:
        """Process the interpolation.

        This is the default implementation and will probably need to be
        overridden in a subclass.
        """
        # Determine if an extra dimension was added to the output
        added_ndim = int(w0.shape[:-1] in ((), (1,)))
        # Return the interpolation and the number of added dimensions
        return interp, added_ndim

    # ------------------------------------------------------------------------

    @overload
    def __call__(
        self,
        F: VectorField,
        w0: gc.AbstractPhaseSpacePosition | gt.BatchVec6,
        t0: gt.FloatQScalar | gt.FloatScalar,
        t1: gt.FloatQScalar | gt.FloatScalar,
        /,
        saveat: SaveT | None = None,
        *,
        units: AbstractUnitSystem,
        interpolated: Literal[False] = False,
    ) -> gc.PhaseSpacePosition: ...

    @overload
    def __call__(
        self,
        F: VectorField,
        w0: gc.AbstractPhaseSpacePosition | gt.BatchVec6,
        t0: gt.FloatQScalar | gt.FloatScalar,
        t1: gt.FloatQScalar | gt.FloatScalar,
        /,
        saveat: SaveT | None = None,
        *,
        units: AbstractUnitSystem,
        interpolated: Literal[True],
    ) -> gc.InterpolatedPhaseSpacePosition: ...

    # TODO: shape hint of the return type
    def __call__(
        self,
        F: VectorField,
        w0: gc.AbstractPhaseSpacePosition | gt.BatchVec6,
        t0: gt.FloatQScalar | gt.FloatScalar,
        t1: gt.FloatQScalar | gt.FloatScalar,
        /,
        saveat: (
            gt.BatchQVecTime | gt.BatchVecTime | gt.QVecTime | gt.VecTime | None
        ) = None,
        *,
        units: AbstractUnitSystem,
        interpolated: Literal[False, True] = False,
    ) -> gc.PhaseSpacePosition | gc.InterpolatedPhaseSpacePosition:
        """Run the integrator.

        Parameters
        ----------
        F : VectorField, positional-only
            The function to integrate.
        w0 : AbstractPhaseSpacePosition | Array[float, (6,)], positional-only
            Initial conditions ``[q, p]``.
        t0, t1 : Quantity, positional-only
            Initial and final times.

        saveat : (Quantity | Array)[float, (T,)] | None, optional
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
        >>> ws = integrator(pot._integrator_F, w0, t0, t1,
        ...                 saveat=ts, units=usx.galactic)
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
        # Parse inputs
        w0_: gt.Vec6 = (
            w0.w(units=units) if isinstance(w0, gc.AbstractPhaseSpacePosition) else w0
        )
        t0_: gt.VecTime = to_units_value(t0, units["time"])
        t1_: gt.VecTime = to_units_value(t1, units["time"])
        # Either save at `saveat` or at the final time. The final time is
        # a scalar and the saveat is a vector, so a dimension is added.
        saveat_ = (
            xp.asarray([t1_])
            if saveat is None
            else to_units_value(saveat, units["time"])
        )

        # Perform the integration
        w, interp = self._call_implementation(F, w0_, t0_, t1_, saveat_, interpolated)
        w = w[..., -1, :] if saveat is None else w  # get rid of added dimension

        # Return
        if interpolated:
            interp, added_ndim = self._process_interpolation(interp, w0_)

            out = gc.InterpolatedPhaseSpacePosition(  # shape = (*batch, T)
                q=Quantity(w[..., 0:3], units["length"]),
                p=Quantity(w[..., 3:6], units["speed"]),
                t=Quantity(saveat_, units["time"]),
                interpolant=self.InterpolantClass(
                    interp, units=units, added_ndim=added_ndim
                ),
            )
        else:
            out = gc.PhaseSpacePosition(  # shape = (*batch, T)
                q=Quantity(w[..., 0:3], units["length"]),
                p=Quantity(w[..., 3:6], units["speed"]),
                t=Quantity(w[..., -1], units["time"]),
            )

        return out

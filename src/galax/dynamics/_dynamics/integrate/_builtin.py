__all__ = ["DiffraxIntegrator"]

from collections.abc import Mapping
from dataclasses import KW_ONLY
from functools import partial
from typing import Any, final

import diffrax
import equinox as eqx
import jax

import quaxed.array_api as xp
from unxt import AbstractUnitSystem, Quantity, to_units_value

import galax.coordinates as gc
import galax.typing as gt
from ._api import FCallable
from ._base import AbstractIntegrator
from galax.utils import ImmutableDict
from galax.utils._jax import vectorize_method


@final
class DiffraxIntegrator(AbstractIntegrator):
    """Integrator using :func:`diffrax.diffeqsolve`.

    This integrator uses the :func:`diffrax.diffeqsolve` function to integrate
    the equations of motion. :func:`diffrax.diffeqsolve` supports a wide range
    of solvers and options. See the documentation of :func:`diffrax.diffeqsolve`
    for more information.

    Parameters
    ----------
    Solver : type[diffrax.AbstractSolver], optional
        The solver to use. Default is :class:`diffrax.Dopri5`.
    stepsize_controller : diffrax.AbstractStepSizeController, optional
        The stepsize controller to use. Default is a PID controller with
        relative and absolute tolerances of 1e-7.
    diffeq_kw : Mapping[str, Any], optional
        Keyword arguments to pass to :func:`diffrax.diffeqsolve`. Default is
        ``{"max_steps": None, "discrete_terminating_event": None}``. The
        ``"max_steps"`` key is removed if ``interpolated=True`` in the
        :meth`DiffraxIntegrator.__call__` method.
    solver_kw : Mapping[str, Any], optional
        Keyword arguments to pass to the solver. Default is ``{"scan_kind":
        "bounded"}``.

    """

    _: KW_ONLY
    Solver: type[diffrax.AbstractSolver] = eqx.field(
        default=diffrax.Dopri5, static=True
    )
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(
        default=diffrax.PIDController(rtol=1e-7, atol=1e-7), static=True
    )
    diffeq_kw: Mapping[str, Any] = eqx.field(
        default=(("max_steps", None), ("discrete_terminating_event", None)),
        static=True,
        converter=ImmutableDict,
    )
    solver_kw: Mapping[str, Any] = eqx.field(
        default=(("scan_kind", "bounded"),), static=True, converter=ImmutableDict
    )

    @vectorize_method(excluded=(0,), signature="(6),(),(),(T)->(T,7)")
    @partial(jax.jit, static_argnums=(0, 1))
    def _call_implementation(
        self,
        F: FCallable,
        w0: gt.Vec6,
        t0: gt.FloatScalar,
        t1: gt.FloatScalar,
        ts: gt.VecTime,
        /,
    ) -> gt.VecTime7:
        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(F),
            solver=self.Solver(**self.solver_kw),
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=None,
            args=(),
            saveat=diffrax.SaveAt(t0=False, t1=False, ts=ts, dense=False),
            stepsize_controller=self.stepsize_controller,
            **self.diffeq_kw,
        )
        ts = solution.ts[:, None] if solution.ts.ndim == 1 else solution.ts
        return xp.concat((solution.ys, ts), axis=1)

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
    ) -> gc.PhaseSpacePosition:
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

        >>> pot = gp.HernquistPotential(m=Quantity(1e12, "Msun"), c=Quantity(5, "kpc"),
        ...                             units="galactic")

        >>> integrator = gd.integrate.DiffraxIntegrator()
        >>> t0, t1 = Quantity(0, "Gyr"), Quantity(1, "Gyr")
        >>> w = integrator(pot._integrator_F, w0, t0, t1, units=usx.galactic)
        >>> w
        PhaseSpacePosition(
            q=Cartesian3DVector( ... ),
            p=CartesianDifferential3D( ... ),
            t=Quantity[...](value=f64[], unit=Unit("Myr"))
        )
        >>> w.shape
        ()

        We can also request the orbit at specific times:

        >>> ts = Quantity(xp.linspace(0, 1, 10), "Myr")  # 10 steps
        >>> ws = integrator(pot._integrator_F, w0, t0, t1, savet=ts, units=usx.galactic)
        >>> ws
        PhaseSpacePosition(
            q=Cartesian3DVector( ... ),
            p=CartesianDifferential3D( ... ),
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
        t0_: gt.VecTime = to_units_value(t0, units["time"])
        t1_: gt.VecTime = to_units_value(t1, units["time"])
        savet_ = (
            xp.asarray([t1_]) if savet is None else to_units_value(savet, units["time"])
        )

        w0_: gt.Vec6 = (
            w0.w(units=units) if isinstance(w0, gc.AbstractPhaseSpacePosition) else w0
        )

        # Perform the integration
        w = self._call_implementation(F, w0_, t0_, t1_, savet_)
        w = w[..., -1, :] if savet is None else w

        # Return
        return gc.PhaseSpacePosition(
            q=Quantity(w[..., 0:3], units["length"]),
            p=Quantity(w[..., 3:6], units["speed"]),
            t=Quantity(w[..., -1], units["time"]),
        )

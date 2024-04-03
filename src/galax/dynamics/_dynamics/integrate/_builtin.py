__all__ = ["DiffraxIntegrator"]

import functools
from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY
from functools import partial
from typing import Any, ParamSpec, TypeVar, final

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jax._src.numpy.vectorize import _parse_gufunc_signature, _parse_input_dimensions

import quaxed.array_api as xp
from unxt import AbstractUnitSystem, Quantity, to_units_value

import galax.coordinates as gc
import galax.typing as gt
from ._api import FCallable
from ._base import AbstractIntegrator
from galax.utils import ImmutableDict

P = ParamSpec("P")
R = TypeVar("R")


def vectorize(
    pyfunc: Callable[P, R], *, signature: str | None = None
) -> Callable[P, R]:
    """Vectorize a function.

    Parameters
    ----------
    pyfunc : Callable[P, R]
        The function to vectorize.
    signature : str | None, optional
        The signature of the vectorized function. Default is `None`.

    Returns
    -------
    Callable[P, R]

    """

    @functools.wraps(pyfunc)
    def wrapped(*args: P.args, **__: P.kwargs) -> R:
        vectorized_func = pyfunc
        input_core_dims, _ = _parse_gufunc_signature(signature)
        broadcast_shape, _ = _parse_input_dimensions(args, input_core_dims, "")

        squeezed_args = []
        rev_filled_shapes = []
        for arg, core_dims in zip(args, input_core_dims, strict=True):
            noncore_shape = jnp.shape(arg)[: jnp.ndim(arg) - len(core_dims)]

            pad_ndim = len(broadcast_shape) - len(noncore_shape)
            filled_shape = pad_ndim * (1,) + noncore_shape
            rev_filled_shapes.append(filled_shape[::-1])

            squeeze_indices = tuple(
                i for i, size in enumerate(noncore_shape) if size == 1
            )
            squeezed_arg = jnp.squeeze(arg, axis=squeeze_indices)
            squeezed_args.append(squeezed_arg)

        for _, axis_sizes in enumerate(zip(*rev_filled_shapes, strict=True)):
            in_axes = tuple(None if size == 1 else 0 for size in axis_sizes)
            if not all(axis is None for axis in in_axes):
                vectorized_func = jax.vmap(vectorized_func, in_axes)

        return vectorized_func(*squeezed_args)

    return wrapped


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

    @partial(jax.jit, static_argnums=(0, 1))
    def _call_implementation(
        self,
        F: FCallable,
        w0: gt.BatchVec6,
        t0: gt.FloatScalar,
        t1: gt.FloatScalar,
        ts: gt.BatchVecTime,
        /,
    ) -> tuple[gt.BatchVecTime7, None]:
        # TODO: less awkward munging of the diffrax API
        kw = dict(self.diffeq_kw)

        terms = diffrax.ODETerm(F)
        solver = self.Solver(**self.solver_kw)

        # TODO: can the vectorize be pushed into diffeqsolve?
        @partial(vectorize, signature="(6),(),(),(T)->()")
        def solve_diffeq(
            w0: gt.Vec6, t0: gt.FloatScalar, t1: gt.FloatScalar, ts: gt.VecTime
        ) -> diffrax.Solution:
            return diffrax.diffeqsolve(
                terms=terms,
                solver=solver,
                t0=t0,
                t1=t1,
                y0=w0,
                dt0=None,
                args=(),
                saveat=diffrax.SaveAt(t0=False, t1=False, ts=ts, dense=False),
                stepsize_controller=self.stepsize_controller,
                **kw,
            )

        # Perform the integration
        solution = solve_diffeq(w0, t0, t1, ts)

        # Parse the solution
        w = jnp.concat((solution.ys, solution.ts[..., None]), axis=-1)
        w = w[None] if w0.shape[0] == 1 else w  # re-add squeezed batch dim
        interp = solution.interpolation

        return w, interp

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

        Instead of just returning the final position, we can get the state of
        the system at any times ``savet``:

        >>> ts = Quantity(xp.linspace(0, 1, 10), "Gyr")  # 10 steps
        >>> ws = integrator(pot._integrator_F, w0, t0, t1, savet=ts, units=usx.galactic)
        >>> ws
        PhaseSpacePosition(
            q=Cartesian3DVector( ... ),
            p=CartesianDifferential3D( ... ),
            t=Quantity[...](value=f64[10], unit=Unit("Myr"))
        )
        >>> ws.shape
        (10,)

        In all these examples the integrator was used to integrate a single
        position. The integrator can also be used to integrate a batch of
        initial conditions at once, returning a batch of final conditions (or a
        batch of conditions at the requested times):

        >>> w0 = gc.PhaseSpacePosition(q=Quantity([[10., 0, 0], [11., 0, 0]], "kpc"),
        ...                            p=Quantity([[0, 200, 0], [0, 210, 0]], "km/s"))
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
        w, interp = self._call_implementation(F, w0_, t0_, t1_, savet_)
        w = w[..., -1, :] if savet is None else w

        # Return
        return gc.PhaseSpacePosition(  # shape = (*batch, T)
            q=Quantity(w[..., 0:3], units["length"]),
            p=Quantity(w[..., 3:6], units["speed"]),
            t=Quantity(w[..., -1], units["time"]),
        )

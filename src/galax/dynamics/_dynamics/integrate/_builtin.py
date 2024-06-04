__all__ = ["DiffraxIntegrator"]

import functools
from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY
from functools import partial
from typing import Any, Literal, ParamSpec, TypeVar, final

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import DenseInterpolation
from jax._src.numpy.vectorize import _parse_gufunc_signature, _parse_input_dimensions
from plum import overload

import quaxed.array_api as xp
from unxt import AbstractUnitSystem, Quantity, to_units_value, unitsystem

import galax.coordinates as gc
import galax.typing as gt
from ._api import FCallable, SaveT
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

    # =====================================================
    # Call

    @partial(eqx.filter_jit)
    def _call_implementation(
        self,
        F: FCallable,
        w0: gt.BatchVec6,
        t0: gt.FloatScalar,
        t1: gt.FloatScalar,
        ts: gt.BatchVecTime,
        /,
        interpolated: Literal[False, True],
    ) -> tuple[gt.BatchVecTime7, DenseInterpolation | None]:
        # TODO: less awkward munging of the diffrax API
        kw = dict(self.diffeq_kw)
        if interpolated and kw.get("max_steps") is None:
            kw.pop("max_steps")

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
                saveat=diffrax.SaveAt(t0=False, t1=False, ts=ts, dense=interpolated),
                stepsize_controller=self.stepsize_controller,
                **kw,
            )

        # Perform the integration
        solution = solve_diffeq(w0, t0, t1, jnp.atleast_2d(ts))

        # Parse the solution
        w = jnp.concat((solution.ys, solution.ts[..., None]), axis=-1)
        w = w[None] if w0.shape[0] == 1 else w  # re-add squeezed batch dim
        interp = solution.interpolation

        return w, interp

    @overload
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
        interpolated: Literal[False] = False,
    ) -> gc.PhaseSpacePosition: ...

    @overload
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
        interpolated: Literal[True],
    ) -> gc.InterpolatedPhaseSpacePosition: ...

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

        Instead of just returning the final position, we can get the state of
        the system at any times ``savet``:

        >>> ts = Quantity(xp.linspace(0, 1, 10), "Gyr")  # 10 steps
        >>> ws = integrator(pot._integrator_F, w0, t0, t1, savet=ts, units=usx.galactic)
        >>> ws
        PhaseSpacePosition(
            q=CartesianPosition3D( ... ),
            p=CartesianVelocity3D( ... ),
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

        A cool feature of the integrator is that it can return an interpolated
        solution.

        >>> w = integrator(pot._integrator_F, w0, t0, t1, savet=ts, units=usx.galactic,
        ...                interpolated=True)
        >>> type(w)
        <class 'galax.coordinates...InterpolatedPhaseSpacePosition'>

        The interpolated solution can be evaluated at any time in the domain to get
        the phase-space position at that time:

        >>> t = Quantity(xp.e, "Gyr")
        >>> w(t)
        PhaseSpacePosition(
            q=CartesianPosition3D( ... ),
            p=CartesianVelocity3D( ... ),
            t=Quantity[PhysicalType('time')](value=f64[1], unit=Unit("Gyr"))
        )

        The interpolant is vectorized:

        >>> t = Quantity(xp.linspace(0, 1, 100), "Gyr")
        >>> w(t)
        PhaseSpacePosition(
            q=CartesianPosition3D( ... ),
            p=CartesianVelocity3D( ... ),
            t=Quantity[PhysicalType('time')](value=f64[1,100], unit=Unit("Gyr"))
        )

        And it works on batches:

        >>> w0 = gc.PhaseSpacePosition(q=Quantity([[10., 0, 0], [11., 0, 0]], "kpc"),
        ...                            p=Quantity([[0, 200, 0], [0, 210, 0]], "km/s"))
        >>> ws = integrator(pot._integrator_F, w0, t0, t1, units=usx.galactic,
        ...                 interpolated=True)
        >>> ws.shape
        (2,)
        >>> w(t)
        PhaseSpacePosition(
            q=CartesianPosition3D( ... ),
            p=CartesianVelocity3D( ... ),
            t=Quantity[PhysicalType('time')](value=f64[1,100], unit=Unit("Gyr"))
        )
        """
        # Parse inputs
        w0_: gt.Vec6 = (
            w0.w(units=units) if isinstance(w0, gc.AbstractPhaseSpacePosition) else w0
        )
        t0_: gt.VecTime = to_units_value(t0, units["time"])
        t1_: gt.VecTime = to_units_value(t1, units["time"])
        savet_ = (
            xp.asarray([t1_]) if savet is None else to_units_value(savet, units["time"])
        )

        # Perform the integration
        w, interp = self._call_implementation(F, w0_, t0_, t1_, savet_, interpolated)
        w = w[..., -1, :] if savet is None else w  # TODO: undo this

        # Return
        if interpolated:
            # Determine if an extra dimension was added to the output
            added_ndim = int(w0_.shape[:-1] in ((), (1,)))
            # If one was, then the interpolant must be reshaped since the input
            # was squeezed beforehand and the dimension must be added back.
            if added_ndim == 1:
                arr, narr = eqx.partition(interp, eqx.is_array)
                arr = jax.tree_util.tree_map(lambda x: x[None], arr)
                interp = eqx.combine(arr, narr)

            out = gc.InterpolatedPhaseSpacePosition(  # shape = (*batch, T)
                q=Quantity(w[..., 0:3], units["length"]),
                p=Quantity(w[..., 3:6], units["speed"]),
                t=Quantity(savet_, units["time"]),
                interpolant=DiffraxInterpolant(
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


class DiffraxInterpolant(eqx.Module):  # type: ignore[misc]#
    """Wrapper for ``diffrax.DenseInterpolation``."""

    interpolant: DenseInterpolation
    """:class:`diffrax.DenseInterpolation` object.

    This object is the result of the integration and can be used to evaluate the
    interpolated solution at any time. However it does not understand units, so
    the input is the time in ``units["time"]``. The output is a 6-vector of
    (q, p) values in the units of the integrator.
    """

    units: AbstractUnitSystem = eqx.field(static=True, converter=unitsystem)
    """The :class:`unxt.AbstractUnitSystem`.

    This is used to convert the time input to the interpolant and the phase-space
    position output.
    """

    added_ndim: gt.Shape = eqx.field(static=True)
    """The number of dimensions added to the output of the interpolation.

    This is used to reshape the output of the interpolation to match the batch
    shape of the input to the integrator. The means of vectorizing the
    interpolation means that the input must always be a batched array, resulting
    in an extra dimension when the integration was on a scalar input.
    """

    # @partial(jax.jit)
    def __call__(self, t: gt.QVecTime, **_: Any) -> gc.PhaseSpacePosition:
        """Evaluate the interpolation."""
        # Parse t
        t_ = jnp.atleast_1d(t.to_units_value(self.units["time"]))
        # t_ = eqx.error_if(t_, xp.any(t_<self.interpolant.t0), "t<t0")  # noqa: ERA001
        # t_ = eqx.error_if(t_, xp.any(t_>self.interpolant.t1), "t>t1")  # noqa: ERA001

        # Evaluate the interpolation
        ys = jax.vmap(lambda s: jax.vmap(s.evaluate)(t_))(self.interpolant)
        extra_dims: int = ys.ndim - 3 + self.added_ndim + (t_.ndim - t.ndim)
        ys = ys[(0,) * extra_dims]

        # Construct and return the result
        return gc.PhaseSpacePosition(
            q=Quantity(ys[..., 0:3], self.units["length"]),
            p=Quantity(ys[..., 3:6], self.units["speed"]),
            t=t,
        )

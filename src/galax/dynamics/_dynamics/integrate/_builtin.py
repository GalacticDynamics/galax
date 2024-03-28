__all__ = ["DiffraxIntegrator"]

from collections.abc import Mapping
from dataclasses import KW_ONLY
from functools import partial
from typing import Any, Literal, final

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import DenseInterpolation
from jaxtyping import Array, Float, Shaped
from plum import overload

import quaxed.array_api as xp
from unxt import AbstractUnitSystem, Quantity, unitsystem

import galax.coordinates as gc
import galax.typing as gt
from ._api import FCallable
from ._base import AbstractIntegrator
from galax.utils import ImmutableDict


@partial(jnp.vectorize, signature="(T,6),(T,1)->(T,7)")
def _broadcast_concat(w: gt.VecTime6, ts: gt.VecTime1, /) -> gt.VecTime7:
    return xp.concat((w, ts), axis=1)


def _to_value(
    x: Shaped[Quantity, "*shape"] | Float[Array, "*shape"], unit: gt.Unit, /
) -> Float[Array, "*shape"]:
    return x.to_value(unit) if isinstance(x, Quantity) else x


@final
class DiffraxIntegrator(AbstractIntegrator):
    """Thin wrapper around ``diffrax.diffeqsolve``."""

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

    @partial(jax.jit, static_argnums=(0, 1, 6))
    def _call_implementation(
        self,
        F: FCallable,
        w0: gt.BatchVec6,
        t0: gt.FloatScalar,
        t1: gt.FloatScalar,
        ts: gt.VecTime,
        /,
        interpolated: Literal[False, True],
    ) -> tuple[gt.BatchVecTime7, DenseInterpolation | None]:
        # TODO: less awkward munging of the diffrax API
        kw = dict(self.diffeq_kw)
        if interpolated and kw["max_steps"] is None:
            kw.pop("max_steps")

        # `jax.numpy.vectorize` only works on arrays, so breaks for
        # `diffrax.Solution` objects. To get around this we emulate the
        # `jax.numpy.vectorize` by manually flattening the batch axes to be
        # just the first axis, vmapping, then reshaping the result.
        terms = diffrax.ODETerm(F)
        solver = self.Solver(**self.solver_kw)

        @partial(jax.vmap, in_axes=(0, 0))
        def solve_diffeq(ts: gt.VecTime, w0: gt.Vec6, /) -> diffrax.Solution:
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

        # Reshape inputs. Need to ensure that the inputs are batched then
        # flattened, so that the vmap'ed `solve` can be applied.
        nt = ts.shape[-1]
        batchw = w0.shape[:-1]
        w0 = jnp.atleast_2d(w0)
        ts = jnp.broadcast_to(jnp.atleast_2d(ts), (*w0.shape[:-1], ts.shape[-1]))

        # Perform the integration
        solution = solve_diffeq(ts.reshape(-1, nt), w0.reshape(-1, 6))

        # Parse the solution
        w = jnp.concat((solution.ys, solution.ts[..., None]), axis=-1)
        interp = solution.interpolation

        # Reshape outputs
        w = w.reshape((*batchw, nt, 7))

        return w, interp

    @overload
    def __call__(
        self,
        F: FCallable,
        w0: gc.AbstractPhaseSpacePosition | gt.BatchVec6,
        /,
        ts: gt.BatchQVecTime | gt.BatchVecTime | gt.QVecTime | gt.VecTime,
        *,
        units: AbstractUnitSystem,
        interpolated: Literal[False] = False,
    ) -> gc.PhaseSpacePosition: ...

    @overload
    def __call__(
        self,
        F: FCallable,
        w0: gc.AbstractPhaseSpacePosition | gt.BatchVec6,
        /,
        ts: gt.BatchQVecTime | gt.BatchVecTime | gt.QVecTime | gt.VecTime,
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
        savet: (
            gt.BatchQVecTime | gt.BatchVecTime | gt.QVecTime | gt.VecTime | None
        ) = None,
        *,
        units: AbstractUnitSystem,
        interpolated: Literal[False, True] = False,
    ) -> gc.PhaseSpacePosition | gc.InterpolatedPhaseSpacePosition:
        # Parse inputs
        t0_: gt.VecTime = _to_value(t0, units["time"])
        t1_: gt.VecTime = _to_value(t1, units["time"])
        savet_ = xp.asarray([t1_]) if savet is None else _to_value(savet, units["time"])

        w0_: gt.Vec6 = (
            w0.w(units=units) if isinstance(w0, gc.AbstractPhaseSpacePosition) else w0
        )
        added_ndim = int(w0_.shape[:-1] == ())

        # Perform the integration
        w = self._call_implementation(F, w0_, t0_, t1_, savet_, interpolated)
        w = w[..., -1, :] if savet is None else w

        # Return
        if interpolated:
            out = gc.InterpolatedPhaseSpacePosition(  # shape = (*batch, T)
                q=Quantity(w[..., 0:3], units["length"]),
                p=Quantity(w[..., 3:6], units["speed"]),
                t=Quantity(ts_, units["time"]),
                interpolation=Interpolation(interp, units=units, added_ndim=added_ndim),
            )
        else:
            out = gc.PhaseSpacePosition(  # shape = (*batch, T)
                q=Quantity(w[..., 0:3], units["length"]),
                p=Quantity(w[..., 3:6], units["speed"]),
                t=Quantity(w[..., -1], units["time"]),
            )

        return out


class Interpolation(eqx.Module):  # type: ignore[misc]
    """Wrapper for ``diffrax.DenseInterpolation``."""

    interpolation: DenseInterpolation
    units: AbstractUnitSystem = eqx.field(static=True, converter=unitsystem)
    added_ndim: tuple[int, ...] = eqx.field(static=True)

    def __call__(self, t: gt.QVecTime, **_: Any) -> gt.BatchVecTime6:
        t_ = jnp.atleast_1d(t.to_value(self.units["time"]))
        ys = jax.vmap(lambda s: jax.vmap(s.evaluate)(t_))(self.interpolation)
        return ys[(0,) * (ys.ndim - self.added_ndim - 1)]

__all__ = ["DiffraxIntegrator"]

from collections.abc import Mapping
from dataclasses import KW_ONLY
from functools import partial
from typing import Any, final

import diffrax
import equinox as eqx
import jax
from jaxtyping import Array, Float, Shaped

import quaxed.array_api as xp
from unxt import AbstractUnitSystem, Quantity

import galax.typing as gt
from ._api import FCallable
from ._base import AbstractIntegrator
from galax.coordinates import AbstractPhaseSpacePosition, PhaseSpacePosition
from galax.utils import ImmutableDict
from galax.utils._jax import vectorize_method


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
        w0: AbstractPhaseSpacePosition | gt.BatchVec6,
        t0: gt.FloatQScalar | gt.FloatScalar,
        t1: gt.FloatQScalar | gt.FloatScalar,
        /,
        savet: (
            gt.BatchQVecTime | gt.BatchVecTime | gt.QVecTime | gt.VecTime | None
        ) = None,
        *,
        units: AbstractUnitSystem,
    ) -> PhaseSpacePosition:
        # Parse inputs
        t0_: gt.VecTime = _to_value(t0, units["time"])
        t1_: gt.VecTime = _to_value(t1, units["time"])
        savet_ = xp.asarray([t1_]) if savet is None else _to_value(savet, units["time"])

        w0_: gt.Vec6 = (
            w0.w(units=units) if isinstance(w0, AbstractPhaseSpacePosition) else w0
        )

        # Perform the integration
        w = self._call_implementation(F, w0_, t0_, t1_, savet_)
        w = w[..., -1, :] if savet is None else w

        # Return
        return PhaseSpacePosition(
            q=Quantity(w[..., 0:3], units["length"]),
            p=Quantity(w[..., 3:6], units["speed"]),
            t=Quantity(w[..., -1], units["time"]),
        )

__all__ = ["DiffraxIntegrator"]

from collections.abc import Mapping
from dataclasses import KW_ONLY
from functools import partial
from typing import Any, final

import diffrax
import equinox as eqx
import jax

import quaxed.array_api as xp
from unxt import Quantity, UnitSystem

import galax.typing as gt
from ._api import FCallable
from ._base import AbstractIntegrator
from galax.coordinates import AbstractPhaseSpacePosition, PhaseSpacePosition
from galax.utils import ImmutableDict
from galax.utils._jax import vectorize_method


@final
class DiffraxIntegrator(AbstractIntegrator):
    """Thin wrapper around ``diffrax.diffeqsolve``."""

    _: KW_ONLY
    Solver: type[diffrax.AbstractSolver] = eqx.field(
        default=diffrax.Dopri5, static=True
    )
    SaveAt: type[diffrax.SaveAt] = eqx.field(default=diffrax.SaveAt, static=True)
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

    @vectorize_method(excluded=(0,), signature="(6),(T)->(T,7)")
    @partial(jax.jit, static_argnums=(0, 1))
    def _call_implementation(
        self, F: FCallable, w0: gt.Vec6, ts: gt.VecTime, /
    ) -> gt.VecTime7:
        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(F),
            solver=self.Solver(**self.solver_kw),
            t0=ts[0],
            t1=ts[-1],
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
        /,
        ts: gt.BatchQVecTime | gt.BatchVecTime | gt.QVecTime | gt.VecTime,
        *,
        units: UnitSystem,
    ) -> PhaseSpacePosition:
        # Parse inputs
        ts_: gt.VecTime = ts.to_value(units["time"]) if isinstance(ts, Quantity) else ts
        w0_: gt.Vec6 = (
            w0.w(units=units) if isinstance(w0, AbstractPhaseSpacePosition) else w0
        )

        # Perform the integration
        w = self._call_implementation(F, w0_, ts_)

        # Return
        return PhaseSpacePosition(
            q=Quantity(w[..., 0:3], units["length"]),
            p=Quantity(w[..., 3:6], units["speed"]),
            t=Quantity(w[..., -1], units["time"]),
        )

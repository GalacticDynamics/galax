__all__ = ["DiffraxIntegrator"]

from collections.abc import Mapping
from dataclasses import KW_ONLY
from typing import Any, final

import array_api_jax_compat as xp
import equinox as eqx
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    Dopri5,
    ODETerm,
    PIDController,
    SaveAt as DiffraxSaveAt,
    diffeqsolve,
)
from jaxtyping import Array, Float

from ._api import FCallable
from ._base import AbstractIntegrator
from galax.typing import Vec6
from galax.utils import ImmutableDict
from galax.utils._jax import vectorize_method


@final
class DiffraxIntegrator(AbstractIntegrator):
    """Thin wrapper around ``diffrax.diffeqsolve``."""

    _: KW_ONLY
    Solver: type[AbstractSolver] = eqx.field(default=Dopri5, static=True)
    SaveAt: type[DiffraxSaveAt] = eqx.field(default=DiffraxSaveAt, static=True)
    stepsize_controller: AbstractStepSizeController = eqx.field(
        default=PIDController(rtol=1e-7, atol=1e-7), static=True
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
    def __call__(
        self, F: FCallable, w0: Vec6, ts: Float[Array, "T"], /
    ) -> Float[Array, "T 7"]:
        solution = diffeqsolve(
            terms=ODETerm(F),
            solver=self.Solver(**self.solver_kw),
            t0=ts[0],
            t1=ts[-1],
            y0=w0,
            dt0=None,
            args=(),
            saveat=DiffraxSaveAt(t0=False, t1=False, ts=ts, dense=False),
            stepsize_controller=self.stepsize_controller,
            **self.diffeq_kw,
        )
        ts = solution.ts[:, None] if solution.ts.ndim == 1 else solution.ts
        return xp.concat((solution.ys, ts), axis=1)

__all__ = ["DiffraxIntegrator"]

from collections.abc import Mapping
from dataclasses import KW_ONLY
from typing import Any

import equinox as eqx
import jax.numpy as xp
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    Dopri5,
    ODETerm,
    PIDController,
    diffeqsolve,
)
from diffrax import SaveAt as DiffraxSaveAt
from jaxtyping import Array, Float

from galax.integrate._base import AbstractIntegrator
from galax.typing import FloatScalar, Vec6
from galax.utils import ImmutableDict


class DiffraxIntegrator(AbstractIntegrator):
    """Thin wrapper around ``diffrax.diffeqsolve``."""

    _: KW_ONLY
    Solver: type[AbstractSolver] = eqx.field(default=Dopri5, static=True)
    SaveAt: type[DiffraxSaveAt] = eqx.field(default=DiffraxSaveAt, static=True)
    stepsize_controller: AbstractStepSizeController = eqx.field(
        default=PIDController(rtol=1e-7, atol=1e-7), static=True
    )
    diffeq_kw: Mapping[str, Any] = eqx.field(
        default=(
            ("max_steps", None),
            ("discrete_terminating_event", None),
        ),
        static=True,
        converter=ImmutableDict,
    )
    solver_kw: Mapping[str, Any] = eqx.field(
        default=(("scan_kind", "bounded"),), static=True, converter=ImmutableDict
    )

    def run(
        self,
        qp0: Vec6,
        t0: FloatScalar,
        t1: FloatScalar,
        ts: Float[Array, "T"] | None,
    ) -> Float[Array, "R 7"]:
        solution = diffeqsolve(
            terms=ODETerm(self.F),
            solver=self.Solver(**self.solver_kw),
            t0=t0,
            t1=t1,
            y0=qp0,
            dt0=None,
            args=(),
            saveat=DiffraxSaveAt(t0=False, t1=True, ts=ts, dense=False),
            stepsize_controller=self.stepsize_controller,
            **self.diffeq_kw,
        )
        ts = solution.ts[:, None] if solution.ts.ndim == 1 else solution.ts
        return xp.concatenate((solution.ys, ts), axis=1)

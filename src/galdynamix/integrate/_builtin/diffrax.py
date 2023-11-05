from __future__ import annotations

__all__ = ["DiffraxIntegrator"]

from dataclasses import KW_ONLY

import equinox as eqx
import jax.typing as jt
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    Dopri5,
    ODETerm,
    PIDController,
    diffeqsolve,
)
from diffrax import SaveAt as DiffraxSaveAt

from galdynamix.integrate._base import Integrator


class DiffraxIntegrator(Integrator):
    _: KW_ONLY
    Solver: AbstractSolver = eqx.field(default=Dopri5, static=True)
    SaveAt: DiffraxSaveAt = eqx.field(default=DiffraxSaveAt, static=True)
    stepsize_controller: AbstractStepSizeController = eqx.field(
        default=PIDController(rtol=1e-7, atol=1e-7), static=True
    )

    def run(
        self, w0: jt.Array, t0: jt.Array, t1: jt.Array, ts: jt.Array | None
    ) -> jt.Array:
        solution = diffeqsolve(
            terms=ODETerm(self.F),
            solver=self.Solver(),
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=None,
            saveat=self.SaveAt(t0=False, t1=True, ts=ts, dense=False),
            stepsize_controller=self.stepsize_controller,
            discrete_terminating_event=None,
            max_steps=None,
        )
        return solution.ys

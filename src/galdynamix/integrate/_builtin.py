from __future__ import annotations

__all__ = ["DiffraxIntegrator"]

from dataclasses import KW_ONLY
from typing import Any

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

from galdynamix.integrate._base import AbstractIntegrator


class DiffraxIntegrator(AbstractIntegrator):
    """Thin wrapper around ``diffrax.diffeqsolve``."""

    _: KW_ONLY
    Solver: AbstractSolver = eqx.field(default=Dopri5, static=True)
    SaveAt: DiffraxSaveAt = eqx.field(default=DiffraxSaveAt, static=True)
    stepsize_controller: AbstractStepSizeController = eqx.field(
        default=PIDController(rtol=1e-7, atol=1e-7), static=True
    )
    diffeq_kw: tuple[tuple[str, Any], ...] = eqx.field(
        default_factory=lambda: (
            ("max_steps", None),
            ("discrete_terminating_event", None),
        ),
        static=True,
    )
    solver_kw: tuple[tuple[str, Any], ...] = eqx.field(
        default_factory=lambda: (("scan_kind", "bounded"),), static=True
    )

    def run(
        self, w0: jt.Array, t0: jt.Array, t1: jt.Array, ts: jt.Array | None
    ) -> jt.Array:
        solution = diffeqsolve(
            terms=ODETerm(self.F),
            solver=self.Solver(**dict(self.solver_kw)),
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=None,
            saveat=self.SaveAt(t0=False, t1=True, ts=ts, dense=False),
            stepsize_controller=self.stepsize_controller,
            **dict(self.diffeq_kw),
        )
        return solution.ys

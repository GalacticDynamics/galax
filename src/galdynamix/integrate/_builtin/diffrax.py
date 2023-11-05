from __future__ import annotations

__all__ = ["DiffraxIntegrator"]


import jax.typing as jt
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve

from galdynamix.integrate._base import Integrator


class DiffraxIntegrator(Integrator):
    def run(
        self, w0: jt.Array, t0: jt.Array, t1: jt.Array, ts: jt.Array | None
    ) -> jt.Array:
        solution = diffeqsolve(
            terms=ODETerm(self.F),
            solver=Dopri5(),
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=None,
            saveat=SaveAt(t0=False, t1=True, ts=ts, dense=False),
            stepsize_controller=PIDController(rtol=1e-7, atol=1e-7),
            discrete_terminating_event=None,
            max_steps=None,
        )
        return solution.ys

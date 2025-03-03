"""Orbit interfacing with `diffrax`.

This is private API.

"""

__all__ = ["compute_orbit"]


from typing import Any

import equinox as eqx
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.coordinates as gc
import galax.potential as gp
from .orbit import Orbit
from galax.dynamics._src.orbit.field_hamiltonian import HamiltonianField
from galax.dynamics._src.orbit.solver import OrbitSolver


@dispatch
@eqx.filter_jit
def compute_orbit(
    field: HamiltonianField | gp.AbstractPotential,
    w0: gc.AbstractPhaseSpaceCoordinate,
    ts: Any,
    /,
    *,
    solver: OrbitSolver | None = None,
    dense: bool = False,
) -> Orbit:
    # Parse inputs
    thefield = field if isinstance(field, HamiltonianField) else HamiltonianField(field)
    solver = OrbitSolver() if solver is None else solver
    units = thefield.units
    ts = jnp.atleast_1d(u.ustrip(AllowValue, units["time"], ts))  # ensure t units

    # Initial integration from `w0.t` to `ts[0]`
    # TODO: use `.init()`, `.run()` instead then can directly pass the state
    soln0 = solver.solve(thefield, w0, ts[0], dense=False, unbatch_time=True)

    # Integrate from `ts[0]` to `ts[-1]`
    if ts.shape == (1,):
        soln = soln0
    else:
        soln = solver.solve(
            thefield,
            soln0.ys,
            ts[0],
            ts[-1],
            saveat=ts,
            dense=dense,
            unbatch_time=True,
            vectorize_interpolation=True,
        )

    # Return the orbit
    return Orbit.from_(soln, frame=w0.frame, units=units)


@dispatch
@eqx.filter_jit
def compute_orbit(
    field: gp.AbstractPotential | HamiltonianField,
    w0: gc.PhaseSpacePosition,
    ts: Any,
    /,
    *,
    solver: OrbitSolver | None = None,
    dense: bool = False,
) -> Orbit:
    # Parse inputs
    thefield = field if isinstance(field, HamiltonianField) else HamiltonianField(field)
    solver = OrbitSolver() if solver is None else solver
    units = thefield.units
    ts = jnp.atleast_1d(u.ustrip(AllowValue, units["time"], ts))  # ensure t units

    # Integrate from `ts[0]` to `ts[-1]`
    soln = solver.solve(
        thefield,
        w0,
        ts[0],
        ts[-1],
        saveat=ts,
        dense=dense,
        unbatch_time=True,
        vectorize_interpolation=True,
    )

    # Return the orbit
    return Orbit.from_(soln, frame=w0.frame, units=thefield.units)

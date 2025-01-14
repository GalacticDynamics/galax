"""Utils for dynamics solvers.

This is private API.

"""

__all__ = ["converter_diffeqsolver", "parse_saveat"]

from typing import Any

import diffrax
from plum import dispatch

import unxt as u
from unxt.quantity import AbstractQuantity

from .diffeqsolver import DiffEqSolver


def converter_diffeqsolver(obj: Any, /) -> DiffEqSolver:
    """Convert to a `DiffEqSolver`.

    Examples
    --------
    >>> import diffrax
    >>> from galax.dynamics._src.solve import DiffEqSolver

    >>> diffeqsolve = DiffEqSolver(solver=diffrax.Dopri5())
    >>> converter_diffeqsolver(diffeqsolve)
    DiffEqSolver(
      solver=Dopri5(scan_kind=None),
      stepsize_controller=ConstantStepSize(),
      adjoint=RecursiveCheckpointAdjoint(checkpoints=None)
    )

    >>> converter_diffeqsolver(diffrax.Dopri5())
    DiffEqSolver(
      solver=Dopri5(scan_kind=None),
      stepsize_controller=ConstantStepSize(),
      adjoint=RecursiveCheckpointAdjoint(checkpoints=None)
    )

    >>> try: converter_diffeqsolver(object())
    ... except TypeError as e: print(e)
    cannot convert <object object at ...> to a `DiffEqSolver`.

    """
    if isinstance(obj, DiffEqSolver):
        out = obj
    elif isinstance(obj, diffrax.AbstractSolver):
        out = DiffEqSolver(solver=obj)
    else:
        msg = f"cannot convert {obj} to a `DiffEqSolver`."
        raise TypeError(msg)
    return out


##############################################################################


@dispatch
def parse_saveat(obj: diffrax.SaveAt, /) -> diffrax.SaveAt:
    """Return the input object.

    Examples
    --------
    >>> import diffrax
    >>> parse_saveat(diffrax.SaveAt(ts=[0, 1, 2, 3]))
    SaveAt(
      subs=SubSaveAt( t0=False, t1=False, ts=i64[4],
                      steps=False, fn=<function save_y> ),
      dense=False,
      solver_state=False,
      controller_state=False,
      made_jump=False
    )

    """
    return obj


@dispatch
def parse_saveat(_: u.AbstractUnitSystem, obj: diffrax.SaveAt, /) -> diffrax.SaveAt:
    """Return the input object.

    Examples
    --------
    >>> import diffrax
    >>> import unxt as u

    >>> units = u.unitsystem("galactic")
    >>> parse_saveat(units, diffrax.SaveAt(ts=[0, 1, 2, 3]))
    SaveAt(
      subs=SubSaveAt( t0=False, t1=False, ts=i64[4],
                      steps=False, fn=<function save_y> ),
      dense=False,
      solver_state=False,
      controller_state=False,
      made_jump=False
    )

    """
    return obj


@dispatch
def parse_saveat(
    units: u.AbstractUnitSystem, ts: AbstractQuantity, /
) -> diffrax.SaveAt:
    """Convert to a `SaveAt`.

    Examples
    --------
    >>> import diffrax
    >>> import unxt as u

    >>> units = u.unitsystem("galactic")
    >>> parse_saveat(units, u.Quantity([0, 1, 2, 3], "Myr"))
    SaveAt(
      subs=SubSaveAt( t0=False, t1=False, ts=i64[4],
                      steps=False, fn=<function save_y> ),
      dense=False,
      solver_state=False,
      controller_state=False,
      made_jump=False
    )

    """
    return diffrax.SaveAt(ts=ts.ustrip(units["time"]))

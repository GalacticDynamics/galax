"""Utils for dynamics solvers.

This is private API.

"""

__all__ = ["converter_diffeqsolver"]

from typing import Any

import diffrax

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

"""Utils for dynamics solvers.

This is private API.

"""

__all__ = ["parse_saveat"]

from dataclasses import replace

import diffrax as dfx
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AbstractQuantity


@dispatch
def parse_saveat(obj: dfx.SaveAt, /, *, dense: bool | None) -> dfx.SaveAt:
    """Return the input object.

    Examples
    --------
    >>> import diffrax as dfx
    >>> parse_saveat(dfx.SaveAt(ts=[0, 1, 2, 3]), dense=True)
    SaveAt(
      subs=SubSaveAt( t0=False, t1=False, ts=i64[4],
                      steps=False, fn=<function save_y> ),
      dense=True,
      solver_state=False,
      controller_state=False,
      made_jump=False
    )

    """
    return obj if dense is None else replace(obj, dense=dense)


@dispatch
def parse_saveat(
    _: u.AbstractUnitSystem, obj: dfx.SaveAt, /, *, dense: bool | None
) -> dfx.SaveAt:
    """Return the input object.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import unxt as u

    >>> units = u.unitsystem("galactic")
    >>> parse_saveat(units, dfx.SaveAt(ts=[0, 1, 2, 3]), dense=True)
    SaveAt(
      subs=SubSaveAt( t0=False, t1=False, ts=i64[4],
                      steps=False, fn=<function save_y> ),
      dense=True,
      solver_state=False,
      controller_state=False,
      made_jump=False
    )

    """
    return obj if dense is None else replace(obj, dense=dense)


@dispatch
def parse_saveat(
    units: u.AbstractUnitSystem, ts: AbstractQuantity, /, *, dense: bool | None
) -> dfx.SaveAt:
    """Convert to a `SaveAt`.

    Examples
    --------
    >>> import unxt as u

    >>> units = u.unitsystem("galactic")

    >>> parse_saveat(units, u.Quantity(0.5, "Myr"), dense=True)
    SaveAt(
      subs=SubSaveAt( t0=False, t1=False, ts=weak_f64[1],
                      steps=False, fn=<function save_y> ),
      dense=True,
      ...
    )

    >>> parse_saveat(units, u.Quantity([0, 1, 2, 3], "Myr"), dense=True)
    SaveAt(
      subs=SubSaveAt( t0=False, t1=False, ts=i64[4],
                      steps=False, fn=<function save_y> ),
      dense=True,
      ...
    )

    """
    return dfx.SaveAt(
        ts=jnp.atleast_1d(ts.ustrip(units["time"])),
        dense=False if dense is None else dense,
    )

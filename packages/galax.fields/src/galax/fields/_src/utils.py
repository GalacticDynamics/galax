"""Utils for field solvers.

Semi-public API: import these from `galax.fields.utils`.

"""

__all__ = [
    "parse_saveat",
]

from dataclasses import replace

from jaxtyping import ArrayLike

import diffrax as dfx
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u

#####################################################################
# Parse SaveAt


@dispatch
def parse_saveat(obj: dfx.SaveAt, /, *, dense: bool | None) -> dfx.SaveAt:
    """Return the input object.

    Examples
    --------
    >>> import diffrax as dfx
    >>> parse_saveat(dfx.SaveAt(ts=[0, 1, 2, 3]), dense=True)
    SaveAt(subs=SubSaveAt(ts=i64[4]), dense=True)

    """
    return obj if dense is None else replace(obj, dense=dense)


@dispatch
def parse_saveat(
    _: u.AbstractUnitSystem | None, obj: dfx.SaveAt, /, *, dense: bool | None
) -> dfx.SaveAt:
    """Return the input object.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import unxt as u

    >>> units = u.unitsystem("galactic")
    >>> parse_saveat(units, dfx.SaveAt(ts=[0, 1, 2, 3]), dense=True)
    SaveAt(subs=SubSaveAt(ts=i64[4]), dense=True)

    """
    return obj if dense is None else replace(obj, dense=dense)


@dispatch
def parse_saveat(
    _: u.AbstractUnitSystem | None,
    ts: ArrayLike | list[ArrayLike],
    /,
    *,
    dense: bool | None,
) -> dfx.SaveAt:
    """Convert to a `SaveAt`.

    Examples
    --------
    >>> import jax.numpy as jnp

    >>> parse_saveat(None, 0.5, dense=True)
    SaveAt(subs=SubSaveAt(ts=weak_f64[1]), dense=True)

    >>> parse_saveat(units, [0, 1, 2, 3], dense=True)
    SaveAt(subs=SubSaveAt(ts=i64[4]), dense=True)

    """
    ts = jnp.atleast_1d(jnp.asarray(ts))
    return dfx.SaveAt(ts=ts, dense=False if dense is None else dense)


@dispatch
def parse_saveat(
    units: u.AbstractUnitSystem, ts: u.AbstractQuantity, /, *, dense: bool | None
) -> dfx.SaveAt:
    """Convert to a `SaveAt`.

    Examples
    --------
    >>> import unxt as u

    >>> units = u.unitsystem("galactic")

    >>> parse_saveat(units, u.Q(0.5, "Myr"), dense=True)
    SaveAt(subs=SubSaveAt(ts=weak_f64[1]), dense=True)

    >>> parse_saveat(units, u.Q([0, 1, 2, 3], "Myr"), dense=True)
    SaveAt(subs=SubSaveAt(ts=i64[4]), dense=True)

    """
    return parse_saveat(units, ts.ustrip(units["time"]), dense=dense)

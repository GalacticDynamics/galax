"""Utils for dynamics solvers.

This is private API.

"""

__all__ = ["parse_saveat", "parse_to_t_y"]

from dataclasses import replace
from typing import Any, TypeAlias

import diffrax as dfx
import equinox as eqx
from jaxtyping import ArrayLike
from plum import convert, dispatch

import coordinax.frames as cxf
import coordinax.vecs as cxv
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt
import galax.coordinates as gc
from . import custom_types as gdt
from galax.potential._src.utils import coord_dispatcher, speed_of_light

#####################################################################
# Parse SaveAt


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
    _: u.AbstractUnitSystem | None, obj: dfx.SaveAt, /, *, dense: bool | None
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
    SaveAt(
      subs=SubSaveAt( t0=False, t1=False, ts=weak_f64[1],
                      steps=False, fn=<function save_y> ),
      dense=True,
      ...
    )

    >>> parse_saveat(units, [0, 1, 2, 3], dense=True)
    SaveAt(
      subs=SubSaveAt( t0=False, t1=False, ts=i64[4],
                      steps=False, fn=<function save_y> ),
      dense=True,
      ...
    )

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
    return parse_saveat(units, ts.ustrip(units["time"]), dense=dense)


#####################################################################
# to y0, t0


OptRefFrame: TypeAlias = cxf.AbstractReferenceFrame | None
UnitSystem: TypeAlias = u.AbstractUnitSystem


@coord_dispatcher.abstract
def parse_to_t_y(
    to_frame: OptRefFrame, *args: Any, **kwargs: Any
) -> tuple[tuple[Any, Any], Any]:  # (x, v), t
    """Convert to initial conditions.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> from galax.dynamics._src.utils import parse_to_t_y

    >>> usys = u.unitsystem("galactic")

    - `jax.Array`:

    >>> xyz = jnp.array([1, 0, 0])
    >>> v_xyz = jnp.array([0, 1, 0])
    >>> t = jnp.array(0)
    >>> parse_to_t_y(None, t, (xyz, v_xyz), ustrip=usys)
    (Array(0, dtype=int64, ...),
     (Array([1., 0., 0.], dtype=float64), Array([0., 1., 0.], dtype=float64)))

    >>> txyz = jnp.array([0, 1, 0, 0])
    >>> parse_to_t_y(None, None, (txyz, v_xyz), ustrip=usys)
    (Array(0, dtype=int64),
     (Array([1., 0., 0.], dtype=float64), Array([0., 1., 0.], dtype=float64)))

    >>> parse_to_t_y(None, t, (txyz, v_xyz), ustrip=usys)
    (Array(0, dtype=int64),
     (Array([1., 0., 0.], dtype=float64), Array([0., 1., 0.], dtype=float64)))

    >>> parse_to_t_y(None, t, xyz, v_xyz, ustrip=usys)
    (Array(0, dtype=int64, ...),
     (Array([1., 0., 0.], dtype=float64), Array([0., 1., 0.], dtype=float64)))

    >>> parse_to_t_y(None, txyz, v_xyz, ustrip=usys)
    (Array(0, dtype=int64),
     (Array([1., 0., 0.], dtype=float64), Array([0., 1., 0.], dtype=float64)))

    >>> parse_to_t_y(None, t, txyz, v_xyz, ustrip=usys)
    (Array(0, dtype=int64),
     (Array([1., 0., 0.], dtype=float64), Array([0., 1., 0.], dtype=float64)))

    >>> parse_to_t_y(None, t, jnp.array([1, 0, 0, 0, 1, 0]), ustrip=usys)
    (Array(0, dtype=int64, ...),
     (Array([1., 0., 0.], dtype=float64), Array([0., 1., 0.], dtype=float64)))

    >>> parse_to_t_y(None, jnp.array([0, 1, 0, 0, 1, 0, 0]), ustrip=usys)
    (Array(0, dtype=int64),
     (Array([1., 0., 0.], dtype=float64), Array([1., 0., 0.], dtype=float64)))

    >>> parse_to_t_y(None, t, jnp.array([0, 1, 0, 0, 1, 0, 0]), ustrip=usys)
    (Array(0, dtype=int64),
     (Array([1., 0., 0.], dtype=float64), Array([1., 0., 0.], dtype=float64)))

    - `unxt.AbstractQuantity`:

    >>> xyz = u.Quantity([1, 0, 0], "kpc")
    >>> v_xyz = u.Quantity([0, 1, 0], "km / s")
    >>> t = u.Quantity(1, "Gyr")

    >>> parse_to_t_y(None, t, (xyz, v_xyz), ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    >>> parse_to_t_y(None, t, xyz, v_xyz, ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    >>> txyz = u.Quantity([0, 1, 0, 0], "kpc")
    >>> parse_to_t_y(None, (txyz, v_xyz), ustrip=usys)
    (Array(0., dtype=float64),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    >>> parse_to_t_y(None, u.Quantity(0, "Gyr"), (txyz, v_xyz), ustrip=usys)
    (Array(0., dtype=float64),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    - `coordinax.vecs.AbstractVector`:

    >>> q = cx.vecs.CartesianPos3D.from_(xyz)
    >>> p = cx.vecs.CartesianVel3D.from_(v_xyz)

    >>> parse_to_t_y(None, t, (q, p), ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    >>> parse_to_t_y(None, t, q, p, ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    >>> qt = cx.vecs.FourVector(q=q, t=t)
    >>> parse_to_t_y(None, (qt, p), ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    >>> parse_to_t_y(None, t, (qt, p), ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    - `coordinax.vecs.Space`:

    >>> space = cx.vecs.Space(length=q, speed=p)
    >>> parse_to_t_y(None, t, space, ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    >>> space = cx.vecs.Space(length=qt, speed=p)
    >>> parse_to_t_y(None, space, ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    >>> parse_to_t_y(None, t, space, ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    - `coordinax.frames.AbstractCoordinate`:

    >>> coord = cx.frames.Coordinate(cx.vecs.Space(length=q, speed=p),
    ...                              frame=gc.frames.simulation_frame)
    >>> parse_to_t_y(None, t, coord, ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    >>> coord = cx.frames.Coordinate(cx.vecs.Space(length=qt, speed=p),
    ...                              frame=gc.frames.simulation_frame)
    >>> parse_to_t_y(None, coord, ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    >>> parse_to_t_y(None, t, coord, ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    - `galax.coordinates.PhaseSpacePosition` (no time):

    >>> psp = gc.PhaseSpacePosition(q, p)
    >>> parse_to_t_y(None, t, psp, ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    - `galax.coordinates.PhaseSpaceCoordinate` (with time):

    >>> psp = gc.PhaseSpaceCoordinate(q, p, t)
    >>> parse_to_t_y(None, psp, ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    >>> parse_to_t_y(None, t, psp, ustrip=usys)
    (Array(1000., dtype=float64, weak_type=True),
     (Array([1., 0., 0.], dtype=float64), Array([0. , 0.00102271, 0. ], dtype=float64)))

    """
    raise NotImplementedError  # pragma: no cover


# --------------------


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame,  # TODO: enable Array-like
    t: gt.BBtLikeSz0 | gt.BBtQuSz0,
    xv: tuple[gdt.BBtQarr, gdt.BBtParr],  # [xyz, v_xyz]
    /,
    *,
    ustrip: UnitSystem,
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    """Convert from tuple of arrays."""
    # Parse inputs to array-ish objects
    t = jnp.asarray(t, dtype=None)
    xyz = jnp.asarray(xv[0], dtype=float)
    v_xyz = jnp.asarray(xv[1], dtype=float)

    # Ensure broadcasting
    xyz, v_xyz = jnp.broadcast_arrays(xyz, v_xyz)

    # Apply frame transformation
    if to_frame is not None:  # TODO: handle velocity xfm
        raise NotImplementedError  # pragma: no cover

    # Strip units
    t = u.ustrip(AllowValue, ustrip["time"], t)

    return t, (xyz, v_xyz)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame,
    t: Any,
    qp: tuple[
        gdt.BBtQ | cxv.AbstractPos3D,
        gdt.BBtP | cxv.AbstractVel3D,
    ],
    /,
    *,
    ustrip: UnitSystem,
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    t = eqx.error_if(t, t is None, "t is None, q does not contain a time")
    q = u.ustrip(ustrip["length"], convert(qp[0], u.Quantity))  # TODO: BareQuantity
    p = u.ustrip(ustrip["speed"], convert(qp[1], u.Quantity))  # TODO: BareQuantity
    return parse_to_t_y(to_frame, t, (q, p), ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame,
    tref: Any,
    qp: tuple[cxv.FourVector, Any],
    /,
    *,
    ustrip: UnitSystem,
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    q, p = qp
    t = u.ustrip(AllowValue, ustrip["time"], q.t)
    t = eqx.error_if(
        t,
        tref is not None
        and jnp.logical_not(
            jnp.array_equal(t, u.ustrip(AllowValue, ustrip["time"], tref))
        ),
        "q.t != tref",
    )
    return parse_to_t_y(to_frame, t, (q.q, p), ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame, qp: tuple[cxv.FourVector, Any], /, *, ustrip: UnitSystem
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    q, p = qp
    t = u.ustrip(AllowValue, ustrip["time"], q.t)
    return parse_to_t_y(to_frame, t, (q.q, p), ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame,
    qt: cxv.FourVector,
    p: cxv.CartesianVel3D,
    /,
    *,
    ustrip: UnitSystem,
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    return parse_to_t_y(to_frame, qt.t, (qt.q, p), ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame,
    tref: Any,
    txv: tuple[
        gt.BBtSz4 | gt.BBtQuSz4,  # txyz
        gt.BBtSz3 | gt.BBtQuSz3,  # v_xyz
    ],
    /,
    *,
    ustrip: UnitSystem,
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    # Parse inputs to t, x, v
    tx, v = txv
    t, x = tx[..., 0], tx[..., 1:]
    # Process Quantity inputs
    if u.quantity.is_any_quantity(t):
        t = t / speed_of_light
    t = u.ustrip(AllowValue, ustrip["time"], t)
    # Compare t with tref
    if tref is not None:
        tref = u.ustrip(AllowValue, ustrip["time"], tref)
        t = eqx.error_if(
            t,
            tref is not None and jnp.logical_not(jnp.array_equal(t, tref)),
            "tx[..., 0] != tref",
        )
    return parse_to_t_y(to_frame, t, (x, v), ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame,
    txv: tuple[
        gt.BBtSz4 | gt.BBtQuSz4,  # txyz
        gt.BBtSz3 | gt.BBtQuSz3,  # v_xyz
    ],
    /,
    *,
    ustrip: UnitSystem,
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    return parse_to_t_y(to_frame, None, txv, ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame,
    tx: gt.BBtSz4 | gt.BBtQuSz4,
    v: gt.BBtSz3 | gt.BBtQuSz3,
    /,
    *,
    ustrip: UnitSystem,
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    t, x = tx[..., 0], tx[..., 1:]
    return parse_to_t_y(to_frame, t, (x, v), ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame, t: Any, xv: gt.BBtSz6, /, *, ustrip: UnitSystem
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    return parse_to_t_y(to_frame, t, (xv[..., :3], xv[..., 3:]), ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame, tref: Any, xv: gt.BBtSz7, /, *, ustrip: UnitSystem
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    return parse_to_t_y(to_frame, tref, (xv[..., 0:4], xv[..., 4:]), ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame, xv: gt.BBtSz7, /, *, ustrip: UnitSystem
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    return parse_to_t_y(to_frame, None, (xv[..., 0:4], xv[..., 4:]), ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame, t: Any, q: Any, p: Any, /, *, ustrip: UnitSystem
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    return parse_to_t_y(to_frame, t, (q, p), ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame, t: Any, qp: cxv.Space, /, *, ustrip: UnitSystem
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    q, p = qp["length"], qp["speed"]
    t = eqx.error_if(
        t,
        t is None and not isinstance(q, cxv.FourVector),
        "t is None, q is not a FourVector",
    )
    return parse_to_t_y(to_frame, t, (q, p), ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame, qp: cxv.Space, /, *, ustrip: UnitSystem
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    q, p = qp["length"], qp["speed"]
    return parse_to_t_y(to_frame, None, (q, p), ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame,
    t: Any,
    coord: cxf.AbstractCoordinate,
    /,
    *,
    ustrip: UnitSystem,
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    return parse_to_t_y(to_frame, t, coord.data, ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame, coord: cxf.AbstractCoordinate, /, *, ustrip: UnitSystem
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    return parse_to_t_y(to_frame, coord.data, ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame, t: Any, w: gc.PhaseSpacePosition, /, *, ustrip: UnitSystem
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    return parse_to_t_y(to_frame, t, (w.q, w.p), ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame,
    tref: Any,
    wt: gc.PhaseSpaceCoordinate,
    /,
    *,
    ustrip: UnitSystem,
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    t = u.ustrip(AllowValue, ustrip["time"], wt.t)
    t = eqx.error_if(
        t,
        tref is not None
        and jnp.logical_not(
            jnp.array_equal(t, u.ustrip(AllowValue, ustrip["time"], tref))
        ),
        "wt.t != tref",
    )
    return parse_to_t_y(to_frame, t, (wt.q, wt.p), ustrip=ustrip)


@coord_dispatcher
def parse_to_t_y(
    to_frame: OptRefFrame, wt: gc.PhaseSpaceCoordinate, /, *, ustrip: UnitSystem
) -> tuple[gt.BBtSz0, gdt.BBtQParr]:
    return parse_to_t_y(to_frame, wt.t, (wt.q, wt.p), ustrip=ustrip)

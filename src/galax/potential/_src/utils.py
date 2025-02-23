"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from typing import Any

import equinox as eqx
import numpy as np
from jax.dtypes import canonicalize_dtype
from plum import Dispatcher, convert

import coordinax as cx
import coordinax.frames as cxf
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue, BareQuantity

import galax._custom_types as gt
import galax.coordinates as gc

speed_of_light = u.quantity.BareQuantity(299_792_458, "m/s")


def parse_dtypes(dtype2: np.dtype, dtype1: Any, /) -> np.dtype | None:
    return (
        dtype2
        if dtype1 is None
        else jnp.promote_types(dtype2, canonicalize_dtype(dtype1))
    )


# ==============================================================================

coord_dispatcher = Dispatcher(warn_redefinition=True)


@coord_dispatcher.abstract
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None, *args: Any, **kwargs: Any
) -> tuple[Any, Any]:  # pos, time
    """Parse input arguments to position & time.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> from galax.potential._src.utils import parse_to_xyz_t

    - `jax.Array`-like:

    >>> xyz = [1, 0, 0]
    >>> t = 0
    >>> parse_to_xyz_t(None, xyz, t, dtype=float)
    (Array([1., 0., 0.], dtype=float64),
     Array(0., dtype=float64))

    - `jax.Array`:

    >>> xyz = jnp.array([1, 0, 0])
    >>> t = jnp.array(0)
    >>> parse_to_xyz_t(None, xyz, t, dtype=float)
    (Array([1., 0., 0.], dtype=float64),
     Array(0., dtype=float64))

    >>> txyz = jnp.array([0, 1, 0, 0])
    >>> parse_to_xyz_t(None, txyz, dtype=float)
    (Array([1., 0., 0.], dtype=float64),
     Array(0., dtype=float64))

    >>> parse_to_xyz_t(None, txyz, t, dtype=float)
    (Array([1., 0., 0.], dtype=float64),
     Array(0., dtype=float64))

    - `unxt.Quantity`:

    >>> q = u.Quantity([1, 0, 0], "kpc")
    >>> t = u.Quantity(1, "Gyr")
    >>> parse_to_xyz_t(None, q, t)
    (Quantity['length'](Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(1, dtype=int64, ...), unit='Gyr'))

    >>> parse_to_xyz_t(None, q, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> tq = u.Quantity([0, 1, 0, 0], "kpc")
    >>> parse_to_xyz_t(None, tq)
    (Quantity['length'](Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(0., dtype=float64), unit='kpc s / m'))

    >>> parse_to_xyz_t(None, tq, u.Quantity(0, "Gyr"))
    (Quantity['length'](Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(0., dtype=float64), unit='kpc s / m'))

    - `coordinax.AbstractVector` objects:

    >>> q = cx.vecs.CartesianPos3D.from_([1, 0, 0], "kpc")
    >>> parse_to_xyz_t(None, q, t)
    (BareQuantity(Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(1, dtype=int64, weak_type=True), unit='Gyr'))

    >>> parse_to_xyz_t(None, q, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> tq = cx.vecs.FourVector(q=q, t=t)
    >>> parse_to_xyz_t(None, tq)
    (BareQuantity(Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(1, dtype=int64, weak_type=True), unit='Gyr'))

    >>> parse_to_xyz_t(None, tq, t)
    (BareQuantity(Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(1, dtype=int64, weak_type=True), unit='Gyr'))

    - `coordinax.Space` objects:

    >>> space = cx.Space(length=q)
    >>> parse_to_xyz_t(None, space, t)
    (BareQuantity(Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(1, dtype=int64, weak_type=True), unit='Gyr'))

    >>> parse_to_xyz_t(None, space, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> space = cx.Space(length=tq)
    >>> parse_to_xyz_t(None, space)
    (BareQuantity(Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(1, dtype=int64, weak_type=True), unit='Gyr'))

    >>> parse_to_xyz_t(None, space, t)
    (BareQuantity(Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(1, dtype=int64, weak_type=True), unit='Gyr'))

    - `coordinax.AbstractCoordinate` objects:

    >>> coord = cx.Coordinate(cx.Space(length=q), frame=gc.frames.simulation_frame)
    >>> parse_to_xyz_t(None, coord, t)
    (BareQuantity(Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(1, dtype=int64, weak_type=True), unit='Gyr'))

    >>> parse_to_xyz_t(None, coord, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> coord = cx.Coordinate(cx.Space(length=tq), frame=gc.frames.simulation_frame)
    >>> parse_to_xyz_t(None, coord)
    (BareQuantity(Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(1, dtype=int64, weak_type=True), unit='Gyr'))

    >>> parse_to_xyz_t(None, coord, t)
    (BareQuantity(Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(1, dtype=int64, weak_type=True), unit='Gyr'))

    - `galax.coordinates.PhaseSpacePosition` objects:

    >>> p = cx.vecs.CartesianVel3D.from_([0, 0, 0], "km/s")
    >>> w = gc.PhaseSpacePosition(q=q, p=p)
    >>> parse_to_xyz_t(None, w, t)
    (BareQuantity(Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(1, dtype=int64, weak_type=True), unit='Gyr'))

    >>> parse_to_xyz_t(None, w, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    - `galax.coordinates.PhaseSpaceCoordinate` objects:

    >>> wt = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)

    >>> parse_to_xyz_t(None, wt)
    (BareQuantity(Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(1, dtype=int64, weak_type=True), unit='Gyr'))

    >>> parse_to_xyz_t(None, wt, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> parse_to_xyz_t(None, wt, t)
    (BareQuantity(Array([1, 0, 0], dtype=int64), unit='kpc'),
     Quantity['time'](Array(1, dtype=int64, weak_type=True), unit='Gyr'))

    """


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    xyz: gt.XYZArrayLike,
    t: gt.BBtLikeSz0,  # TODO: consider also "*#batch 1"
    /,
    *,
    dtype: Any = None,
    ustrip: None = None,  # noqa: ARG001
) -> tuple[gt.BBtSz3, gt.BBtSz0]:
    """Parse input arguments to position & time."""
    # Process the input arguments into arrays
    xyz = jnp.asarray(xyz, dtype=dtype)
    t = jnp.asarray(t, dtype=dtype)

    # The coordinates are assumed to be in the simulation frame and may need to
    # be transformed to the target frame.
    if to_frame is not None:
        op = cxf.frame_transform_op(gc.frames.simulation_frame, to_frame)
        xyz, t = op(xyz, t)

    return xyz, t


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    txyz: gt.BBtLikeSz4,  # Cartesian, in the reference frame
    /,
    *,
    dtype: Any = None,
    ustrip: None = None,
) -> tuple[gt.BBtSz3, gt.BBtSz0]:
    """Parse input argument to position & time."""
    txyz = jnp.asarray(txyz, dtype=dtype)
    return parse_to_xyz_t(
        to_frame, txyz[..., 1:4], txyz[..., 0], dtype=None, ustrip=ustrip
    )


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    txyz: gt.BBtLikeSz4,  # Cartesian, in the reference frame
    t_ref: gt.BBtLikeSz0 | None,
    /,
    *,
    dtype: Any = None,
    ustrip: None = None,
) -> tuple[gt.BBtSz3, gt.BBtSz0]:
    """Parse input argument to position & time."""
    txyz = jnp.asarray(txyz, dtype=dtype)
    t, xyz = txyz[..., 0], txyz[..., 1:4]
    t = eqx.error_if(
        t,
        t_ref is not None and jnp.logical_not(jnp.array_equal(t_ref, t)),
        "t != txyz[..., 0], None",
    )
    return parse_to_xyz_t(to_frame, xyz, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher.multi(
    (cxf.AbstractReferenceFrame | None, gt.BBtQuSz3, gt.BBtQuSz0),
    (cxf.AbstractReferenceFrame | None, gt.BBtQuSz3, gt.BBtSz0 | float | int),
)
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    xyz: gt.BBtQuSz3 | gt.BBtSz3,
    t: gt.BBtQuSz0 | gt.BBtSz0 | float | int,
    /,
    *,
    dtype: Any = None,
    ustrip: u.AbstractUnitSystem | None = None,
) -> tuple[gt.BBtQuSz3 | gt.BBtSz3, gt.BBtQuSz0 | gt.BBtSz0]:
    """Parse input arguments to position & time."""
    xyz = jnp.asarray(xyz, dtype=dtype)
    t = jnp.asarray(t, dtype=dtype)

    if ustrip is not None:
        xyz = u.ustrip(AllowValue, ustrip["length"], xyz)
        t = u.ustrip(AllowValue, ustrip["time"], t)

    # The coordinates are assumed to be in the simulation frame and may need to
    # be transformed to the target frame.
    if to_frame is not None:
        op = cxf.frame_transform_op(gc.frames.simulation_frame, to_frame)
        xyz, t = op(xyz, t)

    return xyz, t


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    txyz: gt.BBtQuSz4,
    /,
    *,
    dtype: Any = None,
    ustrip: u.AbstractUnitSystem | None = None,
) -> tuple[gt.BBtQuSz3 | gt.BBtSz3, gt.BBtQuSz0 | gt.BBtSz0]:
    """Parse input arguments to position & time."""
    ct, xyz = txyz[..., 0], txyz[..., 1:4]
    t = ct / speed_of_light
    return parse_to_xyz_t(to_frame, xyz, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    txyz: gt.BBtQuSz4,
    tref: gt.BBtQuSz0 | None,
    /,
    *,
    dtype: Any = None,
    ustrip: u.AbstractUnitSystem | None = None,
) -> tuple[gt.BBtQuSz3 | gt.BBtSz3, gt.BBtQuSz0 | gt.BBtSz0]:
    """Parse input arguments to position & time."""
    ct, xyz = txyz[..., 0], txyz[..., 1:4]
    t = ct / speed_of_light
    t = eqx.error_if(
        t,
        tref is not None and jnp.logical_not(jnp.array_equal(tref, t)),
        "t != txyz[..., 0], None",
    )
    return parse_to_xyz_t(to_frame, xyz, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    q: cx.vecs.AbstractPos3D,
    t: Any,
    /,
    *,
    dtype: Any = None,
    ustrip: u.AbstractUnitSystem | None = None,
) -> tuple[gt.BBtQuSz3 | gt.BBtSz3, gt.BBtQuSz0 | gt.BBtSz0]:
    """Parse input arguments to position & time."""
    xyz = convert(q.vconvert(cx.CartesianPos3D), BareQuantity)
    return parse_to_xyz_t(to_frame, xyz, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    q4: cx.vecs.FourVector,
    /,
    *,
    dtype: Any = None,
    ustrip: u.AbstractUnitSystem | None = None,
) -> tuple[gt.BBtQuSz3, gt.BBtQuSz0]:
    """Parse input arguments to position & time."""
    return parse_to_xyz_t(to_frame, q4.q, q4.t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    q4: cx.vecs.FourVector,
    tref: gt.BBtQuSz0 | None,
    /,
    *,
    dtype: Any = None,
    ustrip: u.AbstractUnitSystem | None = None,
) -> tuple[gt.BBtQuSz3, gt.BBtQuSz0]:
    """Parse input arguments to position & time."""
    t = q4.t
    t = eqx.error_if(
        t,
        tref is not None and jnp.logical_not(jnp.array_equal(tref, t)),
        "t != q4.t, None",
    )
    return parse_to_xyz_t(to_frame, q4.q, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    space: cx.vecs.Space,
    /,
    *,
    dtype: Any = None,
    ustrip: u.AbstractUnitSystem | None = None,
) -> tuple[gt.BBtQuSz3 | gt.BBtSz3, gt.BBtQuSz0 | gt.BBtSz0]:
    """Parse input arguments to position & time."""
    q = space["length"]
    q = eqx.error_if(q, not isinstance(q, cx.vecs.FourVector), "q is not a FourVector")
    return parse_to_xyz_t(to_frame, q, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    space: cx.vecs.Space,
    t: Any,
    /,
    *,
    dtype: Any = None,
    ustrip: u.AbstractUnitSystem | None = None,
) -> tuple[gt.BBtQuSz3 | gt.BBtSz3, gt.BBtQuSz0 | gt.BBtSz0]:
    """Parse input arguments to position & time."""
    q = space["length"]

    # Case 1: 3D position requires time
    if isinstance(q, cx.vecs.AbstractPos3D):
        t = eqx.error_if(t, t is None, "t is required")
        return parse_to_xyz_t(to_frame, q, t, dtype=dtype, ustrip=ustrip)

    # Case 2: 4D position, time must be equal or None
    if isinstance(q, cx.vecs.FourVector):
        return parse_to_xyz_t(to_frame, q, t, dtype=dtype, ustrip=ustrip)

    msg = f"Unsupported position type: {type(q)}"
    raise TypeError(msg)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    coord: cxf.AbstractCoordinate,
    /,
    *,
    dtype: Any = None,
    ustrip: u.AbstractUnitSystem | None = None,
) -> tuple[gt.BBtQuSz3 | gt.BBtSz3, gt.BBtQuSz0 | gt.BBtSz0]:
    """Parse input arguments to position & time."""
    # Transform to the frame
    # TODO: think about the transformation of the time
    coord = coord.to_frame(gc.frames.simulation_frame if to_frame is None else to_frame)
    # Re-dispatch on the data
    # Now that the data is in the correct frame, we can just parse the data.
    return parse_to_xyz_t(None, coord.data, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    coord: cxf.AbstractCoordinate,
    t: Any,
    /,
    *,
    dtype: Any = None,
    ustrip: u.AbstractUnitSystem | None = None,
) -> tuple[gt.BBtQuSz3 | gt.BBtSz3, gt.BBtQuSz0 | gt.BBtSz0]:
    """Parse input arguments to position & time."""
    # Transform to the frame
    # TODO: think about the transformation of the time
    coord = coord.to_frame(gc.frames.simulation_frame if to_frame is None else to_frame)
    # Re-dispatch on the data
    # Now that the data is in the correct frame, we can just parse the data.
    return parse_to_xyz_t(None, coord.data, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    w: gc.PhaseSpacePosition,
    t: Any,
    /,
    *,
    dtype: Any = None,
    ustrip: u.AbstractUnitSystem | None = None,
) -> tuple[gt.BBtQuSz3, gt.BBtQuSz0]:
    """Parse input arguments to position & time."""
    # Transform to the frame
    # TODO: think about the transformation of the time
    w = w.to_frame(gc.frames.simulation_frame if to_frame is None else to_frame)
    # Re-dispatch on the data
    # Now that the data is in the correct frame, we can just parse the data.
    return parse_to_xyz_t(None, w.q, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    wt: gc.AbstractPhaseSpaceCoordinate,
    t: Any,
    /,
    *,
    dtype: Any = None,
    ustrip: u.AbstractUnitSystem | None = None,
) -> tuple[gt.BBtQuSz3, gt.BBtQuSz0]:
    """Parse input arguments to position & time."""
    # Transform to the frame
    wt = wt.to_frame(gc.frames.simulation_frame if to_frame is None else to_frame)
    # Parse `t`
    t = eqx.error_if(
        jnp.asarray(wt.t, dtype=dtype),
        t is not None and jnp.logical_not(jnp.array_equal(wt.t, t)),
        "t != wt.t, None",
    )
    # Re-dispatch on the data
    # Now that the data is in the correct frame, we can just parse the data.
    return parse_to_xyz_t(None, wt.q, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    wt: gc.AbstractPhaseSpaceCoordinate,
    /,
    *,
    dtype: Any = None,
    ustrip: u.AbstractUnitSystem | None = None,
) -> tuple[gt.BBtQuSz3, gt.BBtQuSz0]:
    """Parse input arguments to position & time."""
    # Transform to the frame
    wt = wt.to_frame(gc.frames.simulation_frame if to_frame is None else to_frame)
    # Re-dispatch on the data
    # Now that the data is in the correct frame, we can just parse the data.
    return parse_to_xyz_t(
        None, wt.q, jnp.asarray(wt.t, dtype=dtype), dtype=dtype, ustrip=ustrip
    )

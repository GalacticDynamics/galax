"""galax: Galactic Dynamix in Jax."""

__all__: tuple[str, ...] = ()

import functools as ft

from jaxtyping import Array, Bool
from typing import Any, TypeAlias, cast

import equinox as eqx
import jax
import numpy as np
import optype as op
from jax.dtypes import canonicalize_dtype
from plum import Dispatcher, convert

import coordinax as cx
import coordinax.frames as cxf
import coordinax.vecs as cxv
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue, BareQuantity

import galax._custom_types as gt
import galax.coordinates as gc

OptUSys: TypeAlias = u.AbstractUnitSystem | None

speed_of_light = u.quantity.BareQuantity(299_792_458, "m/s")


def parse_dtypes(dtype2: np.dtype, dtype1: Any, /) -> np.dtype | None:
    return (
        dtype2
        if dtype1 is None
        else jnp.promote_types(dtype2, canonicalize_dtype(dtype1))
    )


# ==============================================================================


@ft.partial(jax.jit, inline=True)
def safe_sqrt(q2: gt.BBtFloatSz0, /) -> gt.BBtFloatSz0:
    """Square root that stays differentiable where its argument vanishes.

    ``sqrt`` has an infinite derivative at 0, so a radius built as
    ``sqrt(x**2 + y**2 + z**2)`` makes `jax.grad` and `jax.hessian` of every
    potential defined in terms of it NaN at the origin. Offsetting by the
    smallest normal float keeps the derivative finite. The offset is ~1e-308
    (float64), so the value is unchanged for any physically meaningful
    position.

    The offset must be applied to the primal, not just the tangent: recovering
    the correct Hessian of a cored profile at the origin needs ``Phi'(r) / r``
    evaluated at a genuinely non-zero ``r``.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from galax.potential._src.utils import safe_sqrt

    The value matches `jnp.sqrt`:

    >>> safe_sqrt(jnp.asarray(4.0))
    Array(2., dtype=float64)

    but the derivative at zero is finite rather than infinite:

    >>> import jax
    >>> jnp.isfinite(jax.grad(safe_sqrt)(jnp.asarray(0.0)))
    Array(True, dtype=bool, weak_type=True)
    >>> jax.grad(jnp.sqrt)(jnp.asarray(0.0))
    Array(inf, dtype=float64...)

    """
    tiny = jnp.finfo(jnp.promote_types(q2.dtype, float)).tiny
    return jnp.sqrt(q2 + tiny)  # type: ignore[no-any-return]


@ft.partial(jax.jit, inline=True)
def safe_vector_norm(x: gt.BBtSz3, /) -> gt.BBtFloatSz0:
    """`jnp.linalg.vector_norm` over the last axis, differentiable at the origin.

    Values match `jnp.linalg.vector_norm` to within a rounding ulp -- the
    offset is a no-op at any meaningful magnitude, but XLA fuses
    ``sum(square(x))`` differently from `vector_norm`'s scaled algorithm. The
    derivative at the origin is finite rather than NaN. See `safe_sqrt`.

    Unlike `vector_norm`, this overflows for ``|x| > ~1e154`` (float64), which
    no physical position reaches.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from galax.potential._src.utils import safe_vector_norm

    >>> xyz = jnp.asarray([3.0, 4.0, 0.0])
    >>> safe_vector_norm(xyz)
    Array(5., dtype=float64)

    At the origin the gradient is zero -- the value `jnp.linalg.vector_norm`
    cannot give -- rather than NaN:

    >>> import jax
    >>> jax.grad(safe_vector_norm)(jnp.zeros(3))
    Array([0., 0., 0.], dtype=float64)
    >>> jax.grad(jnp.linalg.vector_norm)(jnp.zeros(3))
    Array([nan, nan, nan], dtype=float64)

    """
    return safe_sqrt(jnp.sum(jnp.square(x), axis=-1))  # type: ignore[no-any-return]


@ft.partial(jax.jit, inline=True, static_argnames=("unit",))
def r_spherical(xyz: gt.BBtQorVSz3, unit: Any) -> gt.BBtFloatSz0:
    """Spherical radius.

    Uses `safe_vector_norm` so that the gradient and hessian of potentials
    written in terms of ``r`` are finite at the origin rather than NaN.
    """
    xyz = u.ustrip(AllowValue, unit, xyz)
    r = safe_vector_norm(xyz)
    return r  # type: ignore[no-any-return]


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

    >>> q = u.Q([1, 0, 0], "kpc")
    >>> t = u.Q(1, "Gyr")
    >>> parse_to_xyz_t(None, q, t)
    (Q([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, q, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> tq = u.Q([0, 1, 0, 0], "kpc")
    >>> parse_to_xyz_t(None, tq)
    (Q([1, 0, 0], 'kpc'), Q(0., 'kpc s / m'))

    >>> parse_to_xyz_t(None, tq, u.Q(0, "Gyr"))
    (Q([1, 0, 0], 'kpc'), Q(0., 'kpc s / m'))

    - `coordinax.AbstractVector` objects:

    >>> q = cx.vecs.CartesianPos3D.from_([1, 0, 0], "kpc")
    >>> parse_to_xyz_t(None, q, t)
    (BareQuantity([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, q, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> tq = cx.vecs.FourVector(q=q, t=t)
    >>> parse_to_xyz_t(None, tq)
    (BareQuantity([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, tq, t)
    (BareQuantity([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    - `coordinax.KinematicSpace` objects:

    >>> space = cx.KinematicSpace(length=q)
    >>> parse_to_xyz_t(None, space, t)
    (BareQuantity([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, space, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> space = cx.KinematicSpace(length=tq)
    >>> parse_to_xyz_t(None, space)
    (BareQuantity([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, space, t)
    (BareQuantity([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    - `coordinax.AbstractCoordinate` objects:

    >>> coord = cx.Coordinate(cx.KinematicSpace(length=q),
    ...                       frame=gc.frames.simulation_frame)
    >>> parse_to_xyz_t(None, coord, t)
    (BareQuantity([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, coord, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> coord = cx.Coordinate(cx.KinematicSpace(length=tq),
    ...                       frame=gc.frames.simulation_frame)
    >>> parse_to_xyz_t(None, coord)
    (BareQuantity([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, coord, t)
    (BareQuantity([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    - `galax.coordinates.PhaseSpacePosition` objects:

    >>> p = cx.vecs.CartesianVel3D.from_([0, 0, 0], "km/s")
    >>> w = gc.PhaseSpacePosition(q=q, p=p)
    >>> parse_to_xyz_t(None, w, t)
    (BareQuantity([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, w, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    - `galax.coordinates.PhaseSpaceCoordinate` objects:

    >>> wt = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)

    >>> parse_to_xyz_t(None, wt)
    (BareQuantity([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, wt, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> parse_to_xyz_t(None, wt, t)
    (BareQuantity([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    """


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    xyz: gt.XYZArrayLike,
    t: gt.BBtLikeSz0,  # TODO: consider also "*#batch 1"
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,  # noqa: ARG001
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
    ustrip: OptUSys = None,
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
    ustrip: OptUSys = None,
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
    xyz: gt.BBtQorVSz3,
    t: gt.BBtQorVSz0 | float | int,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
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
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
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
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
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
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
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
    ustrip: OptUSys = None,
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
    ustrip: OptUSys = None,
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
    space: cxv.KinematicSpace,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
    """Parse input arguments to position & time."""
    q = space["length"]
    q = eqx.error_if(q, not isinstance(q, cx.vecs.FourVector), "q is not a FourVector")
    return parse_to_xyz_t(to_frame, q, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    space: cxv.KinematicSpace,
    t: Any,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
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
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
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
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
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
    ustrip: OptUSys = None,
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
    ustrip: OptUSys = None,
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
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQuSz3, gt.BBtQuSz0]:
    """Parse input arguments to position & time."""
    # Transform to the frame
    wt = wt.to_frame(gc.frames.simulation_frame if to_frame is None else to_frame)
    # Re-dispatch on the data
    # Now that the data is in the correct frame, we can just parse the data.
    return parse_to_xyz_t(
        None, wt.q, jnp.asarray(wt.t, dtype=dtype), dtype=dtype, ustrip=ustrip
    )


# ============================================================================
# Moved here from `galax.dynamics._src.utils`: `potential` is the lower
# subpackage of the two that use it, so keeping it in `dynamics` meant
# `potential` importing upward.


def _identity[T](x: T) -> T:
    return x


def _reverse[T](x: op.CanGetitem[Any, T]) -> T:
    return x[::-1]


def cond_reverse[T](pred: Bool[Array, ""], x: T) -> T:
    """Reverse `x` if `pred` is True."""
    return cast("T", jax.lax.cond(pred, _reverse, _identity, x))

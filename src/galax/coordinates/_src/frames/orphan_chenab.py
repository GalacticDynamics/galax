"""Orphan-Chenab reference frame."""

__all__ = ["OrphanChenab"]


from typing import final

from plum import dispatch

import coordinax as cx
import quaxed.numpy as jnp
from coordinax.frames import AbstractReferenceFrame


@final
class OrphanChenab(AbstractReferenceFrame):  # type: ignore[misc]
    """Reference frame centered on the Orphan-Chenab stream.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    Define the frame:

    >>> frame = gc.frames.OrphanChenab()
    >>> frame
    OrphanChenab()

    Build frame transformation operators to/from ICRS:

    >>> icrs = cx.frames.ICRS()

    >>> op = cx.frames.frame_transform_op(icrs, frame)
    >>> op
    GalileanRotation(rotation=f64[3,3])

    >>> cx.frames.frame_transform_op(frame, icrs)
    GalileanRotation(rotation=f64[3,3])

    Transform a position from ICRS to Orphan-Chenab:

    - `unxt.Quantity` Cartesian positions:

    >>> xyz = u.Quantity([1.0, 2.0, 3.0], "kpc")
    >>> op(xyz)
    Quantity[...](Array([-3.29303127,  1.06791461,  1.41968428], dtype=float64), unit='kpc')

    - `coordinax.AbstractVector`:

    >>> q = cx.vecs.CartesianPos3D.from_([1.0, 2.0, 3.0], "kpc")
    >>> print(op(q))
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [-3.293  1.068  1.42 ]>

    >>> q = cx.vecs.SphericalPos(r=u.Quantity(1.0, "kpc"), theta=u.Quantity(45, "deg"), phi=u.Quantity(45, "deg"))
    >>> print(op(q))
    <SphericalPos (r[kpc], theta[rad], phi[rad])
        [1.    1.115 3.097]>

    - `coordinax.Space`:

    >>> space = cx.Space(length=q)
    >>> print(op(space))
    Space({
       'length': <SphericalPos (r[kpc], theta[rad], phi[rad])
           [1.    1.115 3.097]>
    })

    - `coordinax.Coordinate`:

    >>> coord = cx.Coordinate(space, frame=icrs)
    >>> print(coord.to_frame(frame))
    Coordinate(
        data=Space({
           'length': <SphericalPos (r[kpc], theta[rad], phi[rad])
               [1.    1.115 3.097]>
        }),
        frame=OrphanChenab()
    )

    - `galax.coordinates.PhaseSpacePosition`:

    >>> p=u.Quantity([1.0, 2.0, 3.0], "km/s")
    >>> w = gc.PhaseSpacePosition(q=q, p=p, frame=icrs)
    >>> print(w.to_frame(frame))
    PhaseSpacePosition(
        q=<SphericalPos (r[kpc], theta[rad], phi[rad])
            [1.    1.115 3.097]>,
        p=<CartesianVel3D (x[km / s], y[km / s], z[km / s])
            [-3.293  1.068  1.42 ]>,
        frame=OrphanChenab())

    - `galax.coordinates.PhaseSpaceCoordinate`:

    >>> w = gc.PhaseSpaceCoordinate(q=q, p=p, t=u.Quantity(0.0, "Gyr"),
    ...                             frame=icrs)
    >>> print(w.to_frame(frame))
    PhaseSpaceCoordinate(
        q=<SphericalPos (r[kpc], theta[rad], phi[rad])
            [1.    1.115 3.097]>,
        p=<CartesianVel3D (x[km / s], y[km / s], z[km / s])
            [-3.293  1.068  1.42 ]>,
        t=Quantity['time'](Array(0., dtype=float64, weak_type=True), unit='Gyr'),
        frame=OrphanChenab())

    """  # noqa: E501


R_icrs_to_oc = jnp.array(
    [
        [-0.44761231, -0.08785756, -0.88990128],
        [-0.84246097, 0.37511331, 0.38671632],
        [0.29983786, 0.92280606, -0.2419219],
    ]
)


@dispatch
def frame_transform_op(
    from_frame: cx.frames.ICRS,  # noqa: ARG001
    to_frame: OrphanChenab,  # noqa: ARG001
    /,
) -> cx.ops.GalileanRotation:
    return cx.ops.GalileanRotation(R_icrs_to_oc)


@dispatch
def frame_transform_op(
    from_frame: OrphanChenab,  # noqa: ARG001
    to_frame: cx.frames.ICRS,  # noqa: ARG001
    /,
) -> cx.ops.GalileanRotation:
    return cx.ops.GalileanRotation(R_icrs_to_oc.T)

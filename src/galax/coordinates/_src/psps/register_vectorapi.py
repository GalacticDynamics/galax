"""Register PSPs with `coordinax`."""

__all__: list[str] = []

from dataclasses import replace
from typing import Any, cast

import jax.numpy as jnp
from plum import convert
from quax import quaxify

import coordinax as cx
import coordinax.ops as cxo
import unxt as u

from .core import PhaseSpacePosition

batched_matmul = quaxify(jnp.vectorize(jnp.matmul, signature="(3,3),(3)->(3)"))


@cx.frames.AbstractCoordinate.vconvert.dispatch  # type: ignore[misc]
def vconvert(
    self: PhaseSpacePosition,
    position_cls: type[cx.vecs.AbstractPos],
    velocity_cls: type[cx.vecs.AbstractVel] | None = None,
    /,
    **kwargs: Any,
) -> PhaseSpacePosition:
    """Return with the components transformed.

    Parameters
    ----------
    position_cls : type[:class:`~vector.AbstractPos`]
        The target position class.
    velocity_cls : type[:class:`~vector.AbstractVel`], optional
        The target differential class. If `None` (default), the differential
        class of the target position class is used.
    **kwargs
        Additional keyword arguments are passed through to `coordinax.vconvert`.

    Returns
    -------
    w : :class:`~galax.coordinates.PhaseSpacePosition`
        The phase-space position with the components transformed.

    Examples
    --------
    With the following imports:

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    We can create a phase-space position and convert it to a 6-vector:

    >>> psp = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"))
    >>> psp.w(units="galactic")
    Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64, ...)

    We can also convert it to a different representation:

    >>> psp.vconvert(cx.vecs.CylindricalPos)
    PhaseSpacePosition(
        q=CylindricalPos( ... ),
        p=CylindricalVel( ... ),
        frame=SimulationFrame()
    )

    We can also convert it to a different representation with a different
    differential class:

    >>> psp.vconvert(cx.vecs.LonLatSphericalPos, cx.vecs.LonCosLatSphericalVel)
    PhaseSpacePosition( q=LonLatSphericalPos(...),
                        p=LonCosLatSphericalVel(...),
                        frame=SimulationFrame() )
    """
    return cast(
        PhaseSpacePosition,
        cx.vconvert({"q": position_cls, "p": velocity_cls}, self, **kwargs),
    )


######################################################################
# Abstract Operators


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.AbstractOperator,
    x: PhaseSpacePosition,
    /,
) -> PhaseSpacePosition:
    """Apply the operator to a phase-space-time position.

    This method calls the method that operates on
    ``PhaseSpacePosition`` by separating the time component from
    the rest of the phase-space position.  Subclasses can implement that
    method to avoid having to implement for both phase-space-time and
    phase-space positions.  Alternatively, they can implement this method
    directly to avoid redispatching.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> op = cx.ops.GalileanSpatialTranslation(u.Quantity([1, 2, 3], "kpc"))
    >>> op
    GalileanSpatialTranslation(CartesianPos3D( ... ))

    We can then apply the operator to a position:

    >>> pos = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"))
    >>> pos
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        frame=SimulationFrame()
    )

    >>> newpos = op(pos)
    >>> newpos
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        frame=SimulationFrame()
    )

    >>> newpos.q.x
    Quantity(Array(2, dtype=int64), unit='kpc')
    """
    msg = "implement this method in the subclass"
    raise NotImplementedError(msg)


######################################################################
# Composite operators


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.AbstractCompositeOperator, x: PhaseSpacePosition, /
) -> PhaseSpacePosition:
    """Apply the operator to the coordinates."""
    for op in self.operators:
        x = op(x)
    return x


######################################################################
# Galilean spatial translation


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.GalileanSpatialTranslation, psp: PhaseSpacePosition, /
) -> PhaseSpacePosition:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> from dataclasses import replace
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> shift = cx.CartesianPos3D.from_(u.Quantity([1, 1, 1], "kpc"))
    >>> op = cx.ops.GalileanSpatialTranslation(shift)

    >>> psp = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([0, 0, 0], "kpc/Gyr"))

    >>> newpsp = op(psp)
    >>> newpsp.q.x
    Quantity(Array(2, dtype=int64), unit='kpc')

    This spatial translation is time independent.

    >>> psp2 = replace(psp)
    >>> op(psp2).q.x == newpsp.q.x
    Array(True, dtype=bool)

    """
    # Shifting the position and time
    q = self(psp.q)
    # Transforming the momentum. The actual value of momentum is not
    # affected by the translation, however for non-Cartesian coordinates the
    # representation of the momentum in will be different.  First transform
    # the momentum to Cartesian coordinates at the original position. Then
    # transform the momentum back to the original representation, but at the
    # translated position.
    p = psp.p.vconvert(cx.CartesianVel3D, psp.q).vconvert(type(psp.p), q)
    # Reasseble and return
    return replace(psp, q=q, p=p)


######################################################################
# Galilean Rotation


@cxo.AbstractOperator.__call__.dispatch
def call(self: cxo.GalileanRotation, psp: PhaseSpacePosition, /) -> PhaseSpacePosition:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> theta = u.Quantity(45, "deg")
    >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
    ...                  [jnp.sin(theta), jnp.cos(theta),  0],
    ...                  [0,             0,              1]])
    >>> op = cx.ops.GalileanRotation(Rz)

    >>> psp = gc.PhaseSpacePosition(q=u.Quantity([1, 0, 0], "m"),
    ...                             p=u.Quantity([1, 0, 0], "m/s"))

    >>> newpsp = op(psp)

    >>> newpsp.q.x
    Quantity(Array(0.70710678, dtype=float64), unit='m')
    >>> newpsp.q.norm()
    BareQuantity(Array(1., dtype=float64), unit='m')

    >>> newpsp.p.x
    Quantity(Array(0.70710678, dtype=float64), unit='m / s')
    >>> newpsp.p.norm()
    Quantity(Array(1., dtype=float64), unit='m / s')

    """
    # Shifting the position
    q = self(psp.q)
    # Transforming the momentum. The momentum is transformed to Cartesian
    # coordinates at the original position. Then the rotation is applied to
    # the momentum. The momentum is then transformed back to the original
    # representation, but at the rotated position.
    pv = convert(psp.p.vconvert(cx.CartesianVel3D, psp.q), u.Quantity)
    pv = batched_matmul(self.rotation, pv)
    p = cx.CartesianVel3D.from_(pv).vconvert(type(psp.p), q)
    # Reasseble and return
    return replace(psp, q=q, p=p)


######################################################################


@cxo.AbstractOperator.__call__.dispatch(precedence=1)
def call(
    self: cxo.Identity,  # noqa: ARG001
    x: PhaseSpacePosition,
    /,
) -> PhaseSpacePosition:
    """Apply the Identity operation.

    This is the identity operation, which does nothing to the input.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> op = cx.ops.Identity()

    >>> psp = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([0, 0, 0], "kpc/Gyr"))

    >>> op(psp)
    PhaseSpacePosition( q=CartesianPos3D( ... ),
                        p=CartesianVel3D( ... ),
                        frame=SimulationFrame() )
    """
    return x

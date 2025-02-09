"""Base classes for operators on coordinates and potentials."""

__all__: list[str] = []

from dataclasses import replace
from typing import Any, cast

import jax.numpy as jnp
from plum import convert
from quax import quaxify

import coordinax as cx
import coordinax.ops as cxo
import unxt as u

from .base import AbstractPhaseSpaceCoordinate
from .base_single import AbstractBasicPhaseSpaceCoordinate

batched_matmul = quaxify(jnp.vectorize(jnp.matmul, signature="(3,3),(3)->(3)"))


@cx.frames.AbstractCoordinate.vconvert.dispatch  # type: ignore[misc]
def vconvert(
    self: AbstractPhaseSpaceCoordinate,
    position_cls: type[cx.vecs.AbstractPos],
    velocity_cls: type[cx.vecs.AbstractVel] | None = None,
    /,
    **kwargs: Any,
) -> AbstractPhaseSpaceCoordinate:
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

    >>> psp = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                               p=u.Quantity([4, 5, 6], "km/s"),
    ...                               t=u.Quantity(0, "Gyr"))
    >>> psp.w(units="galactic")
    Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64, ...)

    We can also convert it to a different representation:

    >>> psp.vconvert(cx.vecs.CylindricalPos)
    PhaseSpaceCoordinate( q=CylindricalPos(...),
                          p=CylindricalVel(...),
                          t=Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'),
                          frame=SimulationFrame() )

    We can also convert it to a different representation with a different
    differential class:

    >>> psp.vconvert(cx.vecs.LonLatSphericalPos, cx.vecs.LonCosLatSphericalVel)
    PhaseSpaceCoordinate( q=LonLatSphericalPos(...),
                          p=LonCosLatSphericalVel(...),
                          t=Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'),
                          frame=SimulationFrame() )
    """
    return cast(
        AbstractPhaseSpaceCoordinate,
        cx.vconvert({"q": position_cls, "p": velocity_cls}, self, **kwargs),
    )


######################################################################
# Abstract Operators


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.AbstractOperator,
    x: AbstractBasicPhaseSpaceCoordinate,
    /,
) -> AbstractBasicPhaseSpaceCoordinate:
    """Apply the operator to a phase-space-time coordinate.

    This method calls the method that operates on
    ``AbstractBasicPhaseSpaceCoordinate`` by separating the time component from
    the rest of the phase-space coordinate.  Subclasses can implement that
    method to avoid having to implement for both phase-space-time and
    phase-space coordinates.  Alternatively, they can implement this method
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

    We can then apply the operator to a coordinate:

    >>> pos = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                               p=u.Quantity([4, 5, 6], "km/s"),
    ...                               t=u.Quantity(0, "Gyr"))
    >>> pos
    PhaseSpaceCoordinate(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'),
        frame=SimulationFrame()
    )

    >>> newpos = op(pos)
    >>> newpos
    PhaseSpaceCoordinate(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'),
        frame=SimulationFrame()
    )

    >>> newpos.q.x
    Quantity['length'](Array(2, dtype=int64), unit='kpc')

    """
    msg = "implement this method in the subclass"  # pragma: no cover
    raise NotImplementedError(msg)  # pragma: no cover


######################################################################
# Composite operators


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.AbstractCompositeOperator, x: AbstractBasicPhaseSpaceCoordinate, /
) -> AbstractBasicPhaseSpaceCoordinate:
    """Apply the operator to the coordinates."""
    for op in self.operators:
        x = op(x)
    return x


######################################################################
# Galilean spatial translation


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.GalileanSpatialTranslation, psp: AbstractBasicPhaseSpaceCoordinate, /
) -> AbstractBasicPhaseSpaceCoordinate:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> from dataclasses import replace
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> shift = cx.CartesianPos3D.from_(u.Quantity([1, 1, 1], "kpc"))
    >>> op = cx.ops.GalileanSpatialTranslation(shift)

    >>> psp = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                               p=u.Quantity([0, 0, 0], "kpc/Gyr"),
    ...                               t=u.Quantity(0, "Gyr"))

    >>> newpsp = op(psp)
    >>> newpsp.q.x
    Quantity['length'](Array(2, dtype=int64), unit='kpc')

    >>> newpsp.t
    Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr')

    This spatial translation is time independent.

    >>> psp2 = replace(psp, t=u.Quantity(1, "Gyr"))
    >>> op(psp2).q.x == newpsp.q.x
    Array(True, dtype=bool)

    """
    # Shifting the coordinate and time
    q = self(psp.q)
    # Transforming the momentum. The actual value of momentum is not
    # affected by the translation, however for non-Cartesian coordinates the
    # representation of the momentum in will be different.  First transform
    # the momentum to Cartesian coordinates at the original coordinate. Then
    # transform the momentum back to the original representation, but at the
    # translated coordinate.
    p = psp.p.vconvert(cx.CartesianVel3D, psp.q).vconvert(type(psp.p), q)
    # Reasseble and return
    return replace(psp, q=q, p=p)


######################################################################
# Galilean translation


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.GalileanTranslation, psp: AbstractBasicPhaseSpaceCoordinate, /
) -> AbstractBasicPhaseSpaceCoordinate:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> from dataclasses import replace
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> op = cx.ops.GalileanTranslation(u.Quantity([2_000, 1, 1, 1], "kpc"))

    >>> psp = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                               p=u.Quantity([0, 0, 0], "kpc/Gyr"),
    ...                               t=u.Quantity(0, "Gyr"))

    >>> newpsp = op(psp)
    >>> newpsp.q.x
    Quantity['length'](Array(2, dtype=int64), unit='kpc')

    >>> newpsp.t.uconvert("Myr")  # doctest: +SKIP
    Quantity['time'](Array(6.52312755, dtype=float64), unit='Myr')

    This spatial translation is time independent.

    >>> psp2 = replace(psp, t=u.Quantity(1, "Gyr"))
    >>> op(psp2).q.x == newpsp.q.x
    Array(True, dtype=bool)

    But the time translation is not.

    >>> op(psp2).t
    Quantity['time'](Array(1.00652313, dtype=float64, ...), unit='Gyr')

    """
    # TODO: ACCOUNT FOR THE VELOCITY?!?
    # Shifting the coordinate and time
    q, t = self(psp.q, psp.t)
    # Transforming the momentum. The actual value of momentum is not
    # affected by the translation, however for non-Cartesian coordinates the
    # representation of the momentum in will be different.  First transform
    # the momentum to Cartesian coordinates at the original coordinate. Then
    # transform the momentum back to the original representation, but at the
    # translated coordinate.
    p = psp.p.vconvert(cx.CartesianVel3D, psp.q).vconvert(type(psp.p), q)
    # Reasseble and return
    return replace(psp, q=q, p=p, t=t)


######################################################################
# Galilean boost


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.GalileanBoost,
    psp: AbstractBasicPhaseSpaceCoordinate,
    /,
) -> AbstractBasicPhaseSpaceCoordinate:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> from dataclasses import replace
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> op = cx.ops.GalileanBoost(u.Quantity([1, 1, 1], "kpc/Gyr"))

    >>> psp = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                               p=u.Quantity([0, 0, 0], "kpc/Gyr"),
    ...                               t=u.Quantity(1, "Gyr"))

    >>> newpsp = op(psp)
    >>> newpsp.q.x
    Quantity['length'](Array(2, dtype=int64), unit='kpc')

    >>> newpsp.t
    Quantity['time'](Array(1, dtype=int64, ...), unit='Gyr')

    This spatial translation is time dependent.

    >>> psp2 = replace(psp, t=u.Quantity(2, "Gyr"))
    >>> op(psp2).q.x
    Quantity['length'](Array(3, dtype=int64), unit='kpc')

    """
    # TODO: ACCOUNT FOR THE VELOCITY?!?
    # Shifting the coordinate and time
    q, t = self(psp.q, psp.t)
    # Transforming the momentum. The actual value of momentum is not
    # affected by the translation, however for non-Cartesian coordinates the
    # representation of the momentum in will be different.  First transform
    # the momentum to Cartesian coordinates at the original coordinate. Then
    # transform the momentum back to the original representation, but at the
    # translated coordinate.
    p = psp.p.vconvert(cx.CartesianVel3D, psp.q).vconvert(type(psp.p), q)
    # Reasseble and return
    return replace(psp, q=q, p=p, t=t)


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.GalileanRotation, psp: AbstractBasicPhaseSpaceCoordinate, /
) -> AbstractBasicPhaseSpaceCoordinate:
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

    >>> psp = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 0, 0], "m"),
    ...                               p=u.Quantity([1, 0, 0], "m/s"),
    ...                               t=u.Quantity(1, "Gyr"))

    >>> newpsp = op(psp)

    >>> newpsp.q.x
    Quantity['length'](Array(0.70710678, dtype=float64), unit='m')
    >>> newpsp.q.norm()
    BareQuantity(Array(1., dtype=float64), unit='m')

    >>> newpsp.p.x
    Quantity['speed'](Array(0.70710678, dtype=float64), unit='m / s')
    >>> newpsp.p.norm()
    Quantity['speed'](Array(1., dtype=float64), unit='m / s')

    The time is not affected by the rotation.
    >>> newpsp.t
    Quantity['time'](Array(1, dtype=int64, ...), unit='Gyr')
    """
    # Shifting the coordinate and time
    q, t = self(psp.q, psp.t)
    # Transforming the momentum. The momentum is transformed to Cartesian
    # coordinates at the original coordinate. Then the rotation is applied to
    # the momentum. The momentum is then transformed back to the original
    # representation, but at the rotated coordinate.
    pv = convert(psp.p.vconvert(cx.CartesianVel3D, psp.q), u.Quantity)
    pv = batched_matmul(self.rotation, pv)
    p = cx.CartesianVel3D.from_(pv).vconvert(type(psp.p), q)
    # Reasseble and return
    return replace(psp, q=q, p=p, t=t)


######################################################################


@cxo.AbstractOperator.__call__.dispatch(precedence=1)
def call(
    self: cxo.Identity,  # noqa: ARG001
    x: AbstractBasicPhaseSpaceCoordinate,
    /,
) -> AbstractBasicPhaseSpaceCoordinate:
    """Apply the Identity operation.

    This is the identity operation, which does nothing to the input.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> op = cx.ops.Identity()

    >>> psp = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                               p=u.Quantity([0, 0, 0], "kpc/Gyr"),
    ...                               t=u.Quantity(0, "Gyr"))

    >>> op(psp)
    PhaseSpaceCoordinate( q=CartesianPos3D( ... ),
                          p=CartesianVel3D( ... ),
                          t=Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'),
                          frame=SimulationFrame() )
    """
    return x

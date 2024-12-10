"""Base classes for operators on coordinates and potentials."""

__all__: list[str] = []

from dataclasses import replace

import jax.numpy as jnp
from plum import convert
from quax import quaxify

import coordinax as cx
import coordinax.ops as cxo
from unxt import Quantity

from .base_psp import AbstractPhaseSpacePosition

batched_matmul = quaxify(jnp.vectorize(jnp.matmul, signature="(3,3),(3)->(3)"))

######################################################################
# Abstract Operators


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.AbstractOperator,
    x: AbstractPhaseSpacePosition,
    /,
) -> AbstractPhaseSpacePosition:
    """Apply the operator to a phase-space-time position.

    This method calls the method that operates on
    ``AbstractPhaseSpacePosition`` by separating the time component from
    the rest of the phase-space position.  Subclasses can implement that
    method to avoid having to implement for both phase-space-time and
    phase-space positions.  Alternatively, they can implement this method
    directly to avoid redispatching.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import galax.coordinates as gc
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> op = cx.operators.GalileanSpatialTranslation(Quantity([1, 2, 3], "kpc"))
    >>> op
    GalileanSpatialTranslation( translation=CartesianPos3D( ... ) )

    We can then apply the operator to a position:

    >>> pos = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                             p=Quantity([4, 5, 6], "km/s"),
    ...                             t=Quantity(0.0, "Gyr"))
    >>> pos
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("Gyr"))
    )

    >>> newpos = op(pos)
    >>> newpos
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("Gyr"))
    )

    >>> newpos.q.x
    Quantity['length'](Array(2., dtype=float64), unit='kpc')
    """
    msg = "implement this method in the subclass"
    raise NotImplementedError(msg)


######################################################################
# Composite operators


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.AbstractCompositeOperator, x: AbstractPhaseSpacePosition, /
) -> AbstractPhaseSpacePosition:
    """Apply the operator to the coordinates."""
    for op in self.operators:
        x = op(x)
    return x


######################################################################
# Galilean spatial translation


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.GalileanSpatialTranslation, psp: AbstractPhaseSpacePosition, /
) -> AbstractPhaseSpacePosition:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> from dataclasses import replace
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> shift = cx.CartesianPos3D.from_(Quantity([1, 1, 1], "kpc"))
    >>> op = cx.operators.GalileanSpatialTranslation(shift)

    >>> psp = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                             p=Quantity([0, 0, 0], "kpc/Gyr"),
    ...                             t=Quantity(0, "Gyr"))

    >>> newpsp = op(psp)
    >>> newpsp.q.x
    Quantity['length'](Array(2., dtype=float64), unit='kpc')

    >>> newpsp.t
    Quantity['time'](Array(0., dtype=float64), unit='Gyr')

    This spatial translation is time independent.

    >>> psp2 = replace(psp, t=Quantity(1, "Gyr"))
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
    p = psp.p.represent_as(cx.CartesianVel3D, psp.q).represent_as(type(psp.p), q)
    # Reasseble and return
    return replace(psp, q=q, p=p)


######################################################################
# Galilean translation


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.GalileanTranslation, psp: AbstractPhaseSpacePosition, /
) -> AbstractPhaseSpacePosition:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> from dataclasses import replace
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> op = cx.operators.GalileanTranslation(Quantity([2_000, 1, 1, 1], "kpc"))

    >>> psp = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                             p=Quantity([0, 0, 0], "kpc/Gyr"),
    ...                             t=Quantity(0, "Gyr"))

    >>> newpsp = op(psp)
    >>> newpsp.q.x
    Quantity['length'](Array(2., dtype=float64), unit='kpc')

    >>> newpsp.t.uconvert("Myr")  # doctest: +SKIP
    Quantity['time'](Array(6.52312755, dtype=float64), unit='Myr')

    This spatial translation is time independent.

    >>> psp2 = replace(psp, t=Quantity(1, "Gyr"))
    >>> op(psp2).q.x == newpsp.q.x
    Array(True, dtype=bool)

    But the time translation is not.

    >>> op(psp2).t
    Quantity['time'](Array(1.00652313, dtype=float64), unit='Gyr')

    """
    # TODO: ACCOUNT FOR THE VELOCITY?!?
    # Shifting the position and time
    q, t = self(psp.q, psp.t)
    # Transforming the momentum. The actual value of momentum is not
    # affected by the translation, however for non-Cartesian coordinates the
    # representation of the momentum in will be different.  First transform
    # the momentum to Cartesian coordinates at the original position. Then
    # transform the momentum back to the original representation, but at the
    # translated position.
    p = psp.p.represent_as(cx.CartesianVel3D, psp.q).represent_as(type(psp.p), q)
    # Reasseble and return
    return replace(psp, q=q, p=p, t=t)


######################################################################
# Galilean boost


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.GalileanBoost,
    psp: AbstractPhaseSpacePosition,
    /,
) -> AbstractPhaseSpacePosition:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> from dataclasses import replace
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> op = cx.operators.GalileanBoost(Quantity([1, 1, 1], "kpc/Gyr"))

    >>> psp = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                             p=Quantity([0, 0, 0], "kpc/Gyr"),
    ...                             t=Quantity(1, "Gyr"))

    >>> newpsp = op(psp)
    >>> newpsp.q.x
    Quantity['length'](Array(2., dtype=float64), unit='kpc')

    >>> newpsp.t
    Quantity['time'](Array(1., dtype=float64), unit='Gyr')

    This spatial translation is time dependent.

    >>> psp2 = replace(psp, t=Quantity(2, "Gyr"))
    >>> op(psp2).q.x
    Quantity['length'](Array(3., dtype=float64), unit='kpc')

    """
    # TODO: ACCOUNT FOR THE VELOCITY?!?
    # Shifting the position and time
    q, t = self(psp.q, psp.t)
    # Transforming the momentum. The actual value of momentum is not
    # affected by the translation, however for non-Cartesian coordinates the
    # representation of the momentum in will be different.  First transform
    # the momentum to Cartesian coordinates at the original position. Then
    # transform the momentum back to the original representation, but at the
    # translated position.
    p = psp.p.represent_as(cx.CartesianVel3D, psp.q).represent_as(type(psp.p), q)
    # Reasseble and return
    return replace(psp, q=q, p=p, t=t)


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.GalileanRotation, psp: AbstractPhaseSpacePosition, /
) -> AbstractPhaseSpacePosition:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> theta = Quantity(45, "deg")
    >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
    ...                  [jnp.sin(theta), jnp.cos(theta),  0],
    ...                  [0,             0,              1]])
    >>> op = cx.operators.GalileanRotation(Rz)

    >>> psp = gc.PhaseSpacePosition(q=Quantity([1, 0, 0], "m"),
    ...                             p=Quantity([1, 0, 0], "m/s"),
    ...                             t=Quantity(1, "Gyr"))

    >>> newpsp = op(psp)

    >>> newpsp.q.x
    Quantity['length'](Array(0.70710678, dtype=float64), unit='m')
    >>> newpsp.q.norm()
    Quantity['length'](Array(1., dtype=float64), unit='m')

    >>> newpsp.p.d_x
    Quantity['speed'](Array(0.70710678, dtype=float64), unit='m / s')
    >>> newpsp.p.norm()
    Quantity['speed'](Array(1., dtype=float64), unit='m / s')

    The time is not affected by the rotation.
    >>> newpsp.t
    Quantity['time'](Array(1., dtype=float64), unit='Gyr')
    """
    # Shifting the position and time
    q, t = self(psp.q, psp.t)
    # Transforming the momentum. The momentum is transformed to Cartesian
    # coordinates at the original position. Then the rotation is applied to
    # the momentum. The momentum is then transformed back to the original
    # representation, but at the rotated position.
    pv = convert(psp.p.represent_as(cx.CartesianVel3D, psp.q), Quantity)
    pv = batched_matmul(self.rotation, pv)
    p = cx.CartesianVel3D.from_(pv).represent_as(type(psp.p), q)
    # Reasseble and return
    return replace(psp, q=q, p=p, t=t)


######################################################################


@cxo.AbstractOperator.__call__.dispatch(precedence=1)
def call(
    self: cxo.Identity,  # noqa: ARG001
    x: AbstractPhaseSpacePosition,
    /,
) -> AbstractPhaseSpacePosition:
    """Apply the Identity operation.

    This is the identity operation, which does nothing to the input.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> op = cx.operators.Identity()

    >>> psp = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                             p=Quantity([0, 0, 0], "kpc/Gyr"),
    ...                             t=Quantity(0, "Gyr"))

    >>> op(psp)
    PhaseSpacePosition( q=CartesianPos3D( ... ),
                        p=CartesianVel3D( ... ),
                        t=Quantity[...](value=f64[], unit=Unit("Gyr")) )
    """
    return x

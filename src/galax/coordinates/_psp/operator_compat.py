"""Base classes for operators on coordinates and potentials."""

__all__: list[str] = []

from dataclasses import replace

import jax.numpy as jnp
from plum import convert
from quax import quaxify

from coordinax import CartesianDifferential3D
from coordinax.operators import (
    AbstractCompositeOperator,
    AbstractOperator,
    GalileanBoostOperator,
    GalileanRotationOperator,
    GalileanSpatialTranslationOperator,
    GalileanTranslationOperator,
    IdentityOperator,
)
from coordinax.operators._base import op_call_dispatch
from jax_quantity import Quantity

from galax.coordinates._psp.base import AbstractPhaseSpacePositionBase
from galax.coordinates._psp.psp import PhaseSpacePosition
from galax.coordinates._psp.pspt import AbstractPhaseSpaceTimePosition

######################################################################
# Abstract Operators


@op_call_dispatch
def call(
    self: AbstractOperator,  # noqa: ARG001
    x: AbstractPhaseSpacePositionBase,  # noqa: ARG001
    t: Quantity["time"],  # noqa: ARG001
    /,
) -> tuple[AbstractPhaseSpacePositionBase, Quantity["time"]]:
    """Apply the operator to the coordinates.

    Examples
    --------
    >>> from jax_quantity import Quantity
    >>> import galax.coordinates as gc
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> op = cx.operators.GalileanSpatialTranslationOperator(Quantity([1, 2, 3], "kpc"))
    >>> op
    GalileanSpatialTranslationOperator( translation=Cartesian3DVector( ... ) )

    We can then apply the operator to a position:

    >>> pos = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                             p=Quantity([4, 5, 6], "km/s"))
    >>> t = Quantity(0.0, "Gyr")
    >>> pos
    PhaseSpacePosition(
        q=Cartesian3DVector( ... ), p=CartesianDifferential3D( ... ) )

    >>> newpos, newt = op(pos, t)
    >>> newpos, newt
    (PhaseSpacePosition( q=Cartesian3DVector( ... ),
                            p=CartesianDifferential3D( ... ) ),
        Quantity['time'](Array(0., dtype=float64, ...), unit='Gyr'))

    >>> newpos.q.x
    Quantity['length'](Array(2., dtype=float64), unit='kpc')
    """
    msg = "implement this method in the subclass"
    raise NotImplementedError(msg)


@op_call_dispatch
def call(
    self: AbstractOperator, x: AbstractPhaseSpaceTimePosition, /
) -> AbstractPhaseSpaceTimePosition:
    """Apply the operator to a phase-space-time position.

    This method calls the method that operates on
    ``AbstractPhaseSpacePositionBase`` by separating the time component from
    the rest of the phase-space position.  Subclasses can implement that
    method to avoid having to implement for both phase-space-time and
    phase-space positions.  Alternatively, they can implement this method
    directly to avoid redispatching.

    Examples
    --------
    >>> from jax_quantity import Quantity
    >>> import galax.coordinates as gc
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> op = cx.operators.GalileanSpatialTranslationOperator(Quantity([1, 2, 3], "kpc"))
    >>> op
    GalileanSpatialTranslationOperator( translation=Cartesian3DVector( ... ) )

    We can then apply the operator to a position:

    >>> pos = gc.PhaseSpaceTimePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                                 p=Quantity([4, 5, 6], "km/s"),
    ...                                 t=Quantity(0.0, "Gyr"))
    >>> pos
    PhaseSpaceTimePosition(
        q=Cartesian3DVector( ... ),
        p=CartesianDifferential3D( ... ),
        t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("Gyr"))
    )

    >>> newpos = op(pos)
    >>> newpos
    PhaseSpaceTimePosition(
        q=Cartesian3DVector( ... ),
        p=CartesianDifferential3D( ... ),
        t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("Gyr"))
    )

    >>> newpos.q.x
    Quantity['length'](Array(2., dtype=float64), unit='kpc')
    """
    # redispatch on (psp, t)
    psp, t = self(PhaseSpacePosition(q=x.q, p=x.p), x.t)
    return replace(x, q=psp.q, p=psp.p, t=t)


######################################################################
# Composite operators


@op_call_dispatch
def call(
    self: AbstractCompositeOperator,
    x: AbstractPhaseSpacePositionBase,
    t: Quantity["time"],
    /,
) -> tuple[AbstractPhaseSpacePositionBase, Quantity["time"]]:
    """Apply the operator to the coordinates."""
    for op in self.operators:
        x, t = op(x, t)
    return x, t


######################################################################
# Galilean spatial translation


@op_call_dispatch
def call(
    self: GalileanSpatialTranslationOperator,
    psp: AbstractPhaseSpacePositionBase,
    t: Quantity["time"],
    /,
) -> tuple[AbstractPhaseSpacePositionBase, Quantity["time"]]:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> from jax_quantity import Quantity
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> shift = cx.Cartesian3DVector.constructor(Quantity([1, 1, 1], "kpc"))
    >>> op = cx.operators.GalileanSpatialTranslationOperator(shift)

    >>> psp = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                             p=Quantity([0, 0, 0], "kpc/Gyr"))

    >>> t = Quantity(0, "Gyr")
    >>> newpsp, newt = op(psp, t)
    >>> newpsp.q.x
    Quantity['length'](Array(2., dtype=float64), unit='kpc')

    >>> newt
    Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr')

    This spatial translation is time independent.

    >>> op(psp, Quantity(1, "Gyr"))[0].q.x == newpsp.q.x
    Array(True, dtype=bool)

    """
    # Shifting the position and time
    q, t = self(psp.q, t)
    # Transforming the momentum. The actual value of momentum is not
    # affected by the translation, however for non-Cartesian coordinates the
    # representation of the momentum in will be different.  First transform
    # the momentum to Cartesian coordinates at the original position. Then
    # transform the momentum back to the original representation, but at the
    # translated position.
    p = psp.p.represent_as(CartesianDifferential3D, psp.q).represent_as(type(psp.p), q)
    # Reasseble and return
    return (replace(psp, q=q, p=p), t)


######################################################################
# Galilean translation


@op_call_dispatch
def call(
    self: GalileanTranslationOperator,
    psp: AbstractPhaseSpacePositionBase,
    t: Quantity["time"],
    /,
) -> tuple[AbstractPhaseSpacePositionBase, Quantity["time"]]:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> from jax_quantity import Quantity
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> op = cx.operators.GalileanTranslationOperator(Quantity([2_000, 1, 1, 1], "kpc"))

    >>> psp = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                             p=Quantity([0, 0, 0], "kpc/Gyr"))

    >>> t = Quantity(0, "Gyr")
    >>> newpsp, newt = op(psp, t)
    >>> newpsp.q.x
    Quantity['length'](Array(2., dtype=float64), unit='kpc')

    >>> newt.to("Myr")
    Quantity['time'](Array(6.52312755, dtype=float64), unit='Myr')

    This spatial translation is time independent.

    >>> op(psp, Quantity(1, "Gyr"))[0].q.x == newpsp.q.x
    Array(True, dtype=bool)

    But the time translation is not.

    >>> op(psp, Quantity(1, "Gyr"))[1]
    Quantity['time'](Array(1.00652313, dtype=float64), unit='Gyr')

    """
    # Shifting the position and time
    q, t = self(psp.q, t)
    # Transforming the momentum. The actual value of momentum is not
    # affected by the translation, however for non-Cartesian coordinates the
    # representation of the momentum in will be different.  First transform
    # the momentum to Cartesian coordinates at the original position. Then
    # transform the momentum back to the original representation, but at the
    # translated position.
    p = psp.p.represent_as(CartesianDifferential3D, psp.q).represent_as(type(psp.p), q)
    # Reasseble and return
    return (replace(psp, q=q, p=p), t)


######################################################################
# Galilean boost


@op_call_dispatch
def call(
    self: GalileanBoostOperator,
    psp: AbstractPhaseSpacePositionBase,
    t: Quantity["time"],
    /,
) -> tuple[AbstractPhaseSpacePositionBase, Quantity["time"]]:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> from jax_quantity import Quantity
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> op = cx.operators.GalileanBoostOperator(Quantity([1, 1, 1], "kpc/Gyr"))

    >>> psp = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                             p=Quantity([0, 0, 0], "kpc/Gyr"))

    >>> t = Quantity(1, "Gyr")
    >>> newpsp, newt = op(psp, t)
    >>> newpsp.q.x
    Quantity['length'](Array(2., dtype=float64), unit='kpc')

    >>> newt
    Quantity['time'](Array(1, dtype=int64, ...), unit='Gyr')

    This spatial translation is time dependent.

    >>> op(psp, Quantity(2, "Gyr"))[0].q.x
    Quantity['length'](Array(3., dtype=float64), unit='kpc')

    """
    # Shifting the position and time
    q, t = self(psp.q, t)
    # Transforming the momentum. The actual value of momentum is not
    # affected by the translation, however for non-Cartesian coordinates the
    # representation of the momentum in will be different.  First transform
    # the momentum to Cartesian coordinates at the original position. Then
    # transform the momentum back to the original representation, but at the
    # translated position.
    p = psp.p.represent_as(CartesianDifferential3D, psp.q).represent_as(type(psp.p), q)
    # Reasseble and return
    return (replace(psp, q=q, p=p), t)


vec_matmul = quaxify(jnp.vectorize(jnp.matmul, signature="(3,3),(3)->(3)"))


@op_call_dispatch
def call(
    self: GalileanRotationOperator,
    psp: AbstractPhaseSpacePositionBase,
    t: Quantity["time"],
    /,
) -> tuple[AbstractPhaseSpacePositionBase, Quantity["time"]]:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from jax_quantity import Quantity
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> theta = Quantity(45, "deg")
    >>> Rz = xp.asarray([[xp.cos(theta), -xp.sin(theta), 0],
    ...                  [xp.sin(theta), xp.cos(theta),  0],
    ...                  [0,             0,              1]])
    >>> op = cx.operators.GalileanRotationOperator(Rz)

    >>> psp = gc.PhaseSpacePosition(q=Quantity([1, 0, 0], "m"),
    ...                             p=Quantity([1, 0, 0], "m/s"))

    >>> newpsp, newt = op(psp, t)

    >>> newpsp.q.x
    Quantity['length'](Array(0.70710678, dtype=float64), unit='m')
    >>> newpsp.q.norm()
    Quantity['length'](Array(1., dtype=float64), unit='m')

    >>> newpsp.p.d_x
    Quantity['speed'](Array(0.70710678, dtype=float64), unit='m / s')
    >>> newpsp.p.norm()
    Quantity['speed'](Array(1., dtype=float64), unit='m / s')

    The time is not affected by the rotation.
    >>> newt
    Quantity['time'](Array(1, dtype=int64, ...), unit='Gyr')
    """
    # Shifting the position and time
    q, t = self(psp.q, t)
    # Transforming the momentum. The momentum is transformed to Cartesian
    # coordinates at the original position. Then the rotation is applied to
    # the momentum. The momentum is then transformed back to the original
    # representation, but at the rotated position.
    pv = convert(psp.p.represent_as(CartesianDifferential3D, psp.q), Quantity)
    pv = vec_matmul(self.rotation, pv)
    p = CartesianDifferential3D.constructor(pv).represent_as(type(psp.p), q)
    # Reasseble and return
    return (replace(psp, q=q, p=p), t)


######################################################################


@op_call_dispatch(precedence=1)
def call(
    self: IdentityOperator,  # noqa: ARG001
    x: AbstractPhaseSpacePositionBase,
    t: Quantity["time"],
    /,
) -> tuple[AbstractPhaseSpacePositionBase, Quantity["time"]]:
    """Apply the Identity operation.

    This is the identity operation, which does nothing to the input.

    Examples
    --------
    >>> from jax_quantity import Quantity
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> op = cx.operators.IdentityOperator()

    >>> psp = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                             p=Quantity([0, 0, 0], "kpc/Gyr"))

    >>> op(psp, Quantity(0, "Gyr"))
    (PhaseSpacePosition( q=Cartesian3DVector( ... ),
                         p=CartesianDifferential3D( ... ) ),
        Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'))
    """
    return x, t

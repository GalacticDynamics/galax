# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = [
    "GalileanSpatialTranslationOperator",
    "GalileanTranslationOperator",
    "GalileanBoostOperator",
    "GalileanRotationOperator",
    "GalileanOperator",
]


from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, final, overload

import array_api_jax_compat as xp
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Shaped
from plum import convert
from quax import quaxify

from jax_quantity import Quantity
from vector import (
    Abstract3DVector,
    Cartesian3DVector,
    CartesianDifferential3D,
    FourVector,
)

from .base import AbstractOperator, op_call_dispatch
from .composite import AbstractCompositeOperator
from .funcs import simplify_op
from .identity import IdentityOperator
from .sequential import OperatorSequence
from galax.coordinates._psp.base import AbstractPhaseSpacePositionBase
from galax.utils.dataclasses import converter_float_array

if TYPE_CHECKING:
    from typing import Self


class AbstractGalileanOperator(AbstractOperator):
    """Abstract base class for Galilean operators on potentials.

    A potential wrapper is a class that wraps another potential.
    The wrapped potential can be accessed through the `__wrapped__` attribute.
    """


##############################################################################
# Translations


@final
class GalileanSpatialTranslationOperator(AbstractGalileanOperator):
    r"""Operator for Galilean spatial translations.

    The coordinate transform is given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t+s, \mathbf{x} + \mathbf {a})

    Parameters
    ----------
    translation : :class:`vector.Abstract3DVector`
        The spatial translation vector. This parameters accepts either a
        :class:`vector.Abstract3DVector` instance or uses
        :meth:`vector.Cartesian3DVector.constructor` to enable a variety of more
        convenient input types to create a Cartesian vector. See
        :class:`vector.Cartesian3DVector` for details.

    Examples
    --------
    We start with the required imports:

    >>> import array_api_jax_compat as xp
    >>> from jax_quantity import Quantity
    >>> import galax.coordinates.operators as gco

    We can then create a spatial translation operator:

    >>> shift = Quantity([1.0, 2.0, 3.0], "kpc")
    >>> op = gco.GalileanSpatialTranslationOperator(shift)
    >>> op
    GalileanSpatialTranslationOperator( translation=Cartesian3DVector( ... ) )

    Note that the translation is a :class:`vector.Cartesian3DVector`, which was
    constructed from a 1D array, using
    :meth:`vector.Cartesian3DVector.constructor`. We can also construct it
    directly, which allows for other vector types.

    >>> from vector import SphericalVector
    >>> shift = SphericalVector(r=Quantity(1.0, "kpc"),
    ...                         theta=Quantity(xp.pi/2, "rad"),
    ...                         phi=Quantity(0, "rad"))
    >>> op = gco.GalileanSpatialTranslationOperator(shift)
    >>> op
    GalileanSpatialTranslationOperator( translation=SphericalVector( ... ) )

    Translation operators can be applied to :class:`vector.AbstractVector`:

    >>> from vector import Cartesian3DVector
    >>> q = Cartesian3DVector.constructor(Quantity([0, 0, 0], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> op(q, t)
    (Cartesian3DVector( ... ), Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'))

    """

    translation: Abstract3DVector = eqx.field(
        converter=lambda x: (
            x if isinstance(x, Abstract3DVector) else Cartesian3DVector.constructor(x)
        )
    )
    """The spatial translation.

    This parameters accepts either a :class:`vector.Abstract3DVector` instance
    or uses :meth:`vector.Cartesian3DVector.constructor` to enable a variety of
    more convenient input types to create a Cartesian vector. See
    :class:`vector.Cartesian3DVector` for details.
    """

    # -------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean translation is an inertial frame-preserving transformation.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector
        >>> from galax.coordinates.operators import GalileanSpatialTranslationOperator

        >>> shift = Cartesian3DVector.constructor(Quantity([1, 1, 1], "kpc"))
        >>> op = GalileanSpatialTranslationOperator(shift)

        >>> op.is_inertial
        True
        """
        return True

    @property
    def inverse(self) -> "GalileanSpatialTranslationOperator":
        """The inverse of the operator.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector
        >>> from galax.coordinates.operators import GalileanSpatialTranslationOperator

        >>> shift = Cartesian3DVector.constructor(Quantity([1, 1, 1], "kpc"))
        >>> op = GalileanSpatialTranslationOperator(shift)

        >>> op.inverse
        GalileanSpatialTranslationOperator( translation=Cartesian3DVector( ... ) )

        >>> op.inverse.translation.x
        Quantity['length'](Array(-1., dtype=float64), unit='kpc')
        """
        return GalileanSpatialTranslationOperator(-self.translation)

    # -------------------------------------------

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanSpatialTranslationOperator",
        q: Abstract3DVector,
        t: Quantity["time"],
        /,
    ) -> tuple[Abstract3DVector, Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector
        >>> from galax.coordinates.operators import GalileanSpatialTranslationOperator

        >>> shift = Cartesian3DVector.constructor(Quantity([1, 1, 1], "kpc"))
        >>> op = GalileanSpatialTranslationOperator(shift)

        >>> q = Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> newq, newt = op(q, t)
        >>> newq.x
        Quantity['length'](Array(2., dtype=float64), unit='kpc')

        >>> newt
        Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr')

        This spatial translation is time independent.

        >>> op(q, Quantity(1, "Gyr"))[0].x == newq.x
        Array(True, dtype=bool)

        """
        return (q + self.translation, t)

    @op_call_dispatch
    def __call__(
        self: "GalileanSpatialTranslationOperator",
        psp: AbstractPhaseSpacePositionBase,
        t: Quantity["time"],
        /,
    ) -> tuple[AbstractPhaseSpacePositionBase, Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector, CartesianDifferential3D
        >>> from galax.coordinates import PhaseSpacePosition
        >>> from galax.coordinates.operators import GalileanSpatialTranslationOperator

        >>> shift = Cartesian3DVector.constructor(Quantity([1, 1, 1], "kpc"))
        >>> op = GalileanSpatialTranslationOperator(shift)

        >>> psp = PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                          p=Quantity([0, 0, 0], "kpc/Gyr"))

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
        p = psp.p.represent_as(CartesianDifferential3D, psp.q).represent_as(
            type(psp.p), q
        )
        # Reasseble and return
        return (replace(psp, q=q, p=p), t)


@final
class GalileanTranslationOperator(AbstractGalileanOperator):
    r"""Operator for spatio-temporal translations.

    The coordinate transform is given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t+s, \mathbf{x} + \mathbf {a})

    where :math:`a \in R^3` and :math:`s \in R`.  Therefore for a potential
    :math:`\Phi(t,\mathbf{x})` in the translated frame the potential is given by
    the subtraction of the translation.

    Parameters
    ----------
    translation : :class:`vector.FourVector`
        The translation vector [T, Q].  This parameters uses
        :meth:`vector.FourVector.constructor` to enable a variety of more
        convenient input types. See :class:`vector.FourVector` for details.

    Examples
    --------
    We start with the required imports:

    >>> import array_api_jax_compat as xp
    >>> from jax_quantity import Quantity
    >>> import galax.coordinates.operators as gco

    We can then create a translation operator:

    >>> op = GalileanTranslationOperator(Quantity([1.0, 2.0, 3.0, 4.0], "kpc"))
    >>> op
    GalileanTranslationOperator(
      translation=FourVector(
        t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("kpc s / km")),
        q=Cartesian3DVector( ... ) )
    )

    Note that the translation is a :class:`vector.FourVector`, which was
    constructed from a 1D array, using :meth:`vector.FourVector.constructor`. We
    can also construct it directly, which allows for other vector types.

    >>> from vector import SphericalVector
    >>> qshift = SphericalVector(r=Quantity(1.0, "kpc"), theta=Quantity(xp.pi/2, "rad"),
    ...                          phi=Quantity(0, "rad"))
    >>> op = GalileanTranslationOperator(FourVector(t=Quantity(1.0, "Gyr"), q=qshift))
    >>> op
    GalileanTranslationOperator(
      translation=FourVector(
        t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("Gyr")),
        q=SphericalVector( ... ) )
    )

    Translation operators can be applied to :class:`vector.FourVector`:

    >>> w = FourVector.constructor(Quantity([0, 0, 0, 0], "kpc"))
    >>> op(w)
    FourVector(
      t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("kpc s / km")),
      q=Cartesian3DVector( ... )
    )

    Also to :class:`vector.Abstract3DVector` and :class:`jax_quantity.Quantity`:

    >>> q = Cartesian3DVector.constructor(Quantity([0, 0, 0], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> newq, newt = op(q, t)
    >>> newq.x
    Quantity['length'](Array(1., dtype=float64), unit='kpc')
    >>> newt
    Quantity['time'](Array(1., dtype=float64), unit='Gyr')
    """

    translation: FourVector = eqx.field(converter=FourVector.constructor)
    """The temporal + spatial translation.

    The translation vector [T, Q].  This parameters uses
    :meth:`vector.FourVector.constructor` to enable a variety of more convenient
    input types. See :class:`vector.FourVector` for details.
    """

    # -------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean translation is an inertial-frame preserving transformation.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import FourVector
        >>> from galax.coordinates.operators import GalileanTranslationOperator

        >>> shift = FourVector.constructor(Quantity([0, 1, 1, 1], "kpc"))
        >>> op = GalileanTranslationOperator(shift)

        >>> op.is_inertial
        True
        """
        return True

    @property
    def inverse(self) -> "GalileanTranslationOperator":
        """The inverse of the operator.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector, FourVector
        >>> from galax.coordinates.operators import GalileanSpatialTranslationOperator

        >>> qshift = Cartesian3DVector.constructor(Quantity([1, 1, 1], "kpc"))
        >>> tshift = Quantity(1, "Gyr")
        >>> shift = FourVector(tshift, qshift)
        >>> op = GalileanTranslationOperator(shift)

        >>> op.inverse
        GalileanTranslationOperator( translation=FourVector( ... ) )

        >>> op.inverse.translation.q.x
        Quantity['length'](Array(-1., dtype=float64), unit='kpc')
        """
        return GalileanTranslationOperator(-self.translation)

    # -------------------------------------------

    @op_call_dispatch
    def __call__(self: "GalileanTranslationOperator", x: FourVector, /) -> FourVector:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector, FourVector
        >>> from galax.coordinates.operators import GalileanTranslationOperator

        Explicitly construct the translation operator:

        >>> qshift = Cartesian3DVector.constructor(Quantity([1, 1, 1], "kpc"))
        >>> tshift = Quantity(1, "Gyr")
        >>> shift = FourVector(tshift, qshift)
        >>> op = GalileanTranslationOperator(shift)

        Construct a vector to translate, using the convenience constructor (the
        0th component is :math:`c * t`, the rest are spatial components):

        >>> w = FourVector.constructor(Quantity([0, 1, 2, 3], "kpc"))
        >>> w.t
        Quantity['time'](Array(0., dtype=float64), unit='kpc s / km')

        Apply the translation operator:

        >>> new = op(w)
        >>> new.x
        Quantity['length'](Array(2., dtype=float64), unit='kpc')

        >>> new.t.to("Gyr")
        Quantity['time'](Array(1., dtype=float64), unit='Gyr')

        """
        return x + self.translation

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanTranslationOperator", x: Abstract3DVector, t: Quantity["time"], /
    ) -> tuple[Abstract3DVector, Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector, FourVector
        >>> from galax.coordinates.operators import GalileanTranslationOperator

        Explicitly construct the translation operator:

        >>> qshift = Cartesian3DVector.constructor(Quantity([1, 1, 1], "kpc"))
        >>> tshift = Quantity(1, "Gyr")
        >>> shift = FourVector(tshift, qshift)
        >>> op = GalileanTranslationOperator(shift)

        Construct a vector to translate

        >>> q = Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(1, "Gyr")
        >>> newq, newt = op(q, t)

        >>> newq.x
        Quantity['length'](Array(2., dtype=float64), unit='kpc')

        >>> newt
        Quantity['time'](Array(2., dtype=float64), unit='Gyr')
        """
        return (x + self.translation.q, t + self.translation.t)

    @op_call_dispatch
    def __call__(
        self: "GalileanTranslationOperator",
        psp: AbstractPhaseSpacePositionBase,
        t: Quantity["time"],
        /,
    ) -> tuple[AbstractPhaseSpacePositionBase, Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector
        >>> from galax.coordinates.operators import GalileanTranslationOperator

        >>> op = GalileanTranslationOperator(Quantity([2_000, 1, 1, 1], "kpc"))

        >>> psp = PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                          p=Quantity([0, 0, 0], "kpc/Gyr"))

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
        p = psp.p.represent_as(CartesianDifferential3D, psp.q).represent_as(
            type(psp.p), q
        )
        # Reasseble and return
        return (replace(psp, q=q, p=p), t)


##############################################################################
# Boosts


@final
class GalileanBoostOperator(AbstractGalileanOperator):
    r"""Operator for Galilean boosts.

    The coordinate transform is given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t, \mathbf{x} + \mathbf{v} t)

    where :math:`\mathbf{v}` is the boost velocity.

    Parameters
    ----------
    velocity : :class:`vector.CartesianDifferential3D`
        The boost velocity. This parameters uses
        :meth:`vector.CartesianDifferential3D.constructor` to enable a variety
        of more convenient input types. See
        :class:`vector.CartesianDifferential3D` for details.

    Examples
    --------
    We start with the required imports:

    >>> import array_api_jax_compat as xp
    >>> from jax_quantity import Quantity
    >>> from vector import CartesianDifferential3D, Cartesian3DVector
    >>> import galax.coordinates.operators as gco

    We can then create a boost operator:

    >>> op = gco.GalileanBoostOperator(Quantity([1.0, 2.0, 3.0], "m/s"))
    >>> op
    GalileanBoostOperator( velocity=CartesianDifferential3D( ... ) )

    Note that the velocity is a :class:`vector.CartesianDifferential3D`, which
    was constructed from a 1D array, using
    :meth:`vector.CartesianDifferential3D.constructor`. We can also construct it
    directly:

    >>> boost = CartesianDifferential3D(d_x=Quantity(1, "m/s"), d_y=Quantity(2, "m/s"),
    ...                                 d_z=Quantity(3, "m/s"))
    >>> op = gco.GalileanBoostOperator(boost)
    >>> op
    GalileanBoostOperator( velocity=CartesianDifferential3D( ... ) )

    Translation operators can be applied to :class:`vector.Abstract3DVector`:

    >>> q = Cartesian3DVector.constructor(Quantity([0, 0, 0], "m"))
    >>> t = Quantity(1, "s")
    >>> newq, newt = op(q, t)
    >>> newt
    Quantity['time'](Array(1, dtype=int64, ...), unit='s')
    >>> newq.x
    Quantity['length'](Array(1., dtype=float64), unit='m')

    """

    velocity: CartesianDifferential3D = eqx.field(
        converter=CartesianDifferential3D.constructor
    )
    """The boost velocity.

    This parameters uses :meth:`vector.CartesianDifferential3D.constructor` to
    enable a variety of more convenient input types. See
    :class:`vector.CartesianDifferential3D` for details.
    """

    # -----------------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean boost is an inertial-frame preserving transform.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector, CartesianDifferential3D
        >>> from galax.coordinates.operators import GalileanBoostOperator

        >>> op = GalileanBoostOperator(Quantity([1, 2, 3], "m/s"))
        >>> op.is_inertial
        True
        """
        return True

    @property
    def inverse(self) -> "GalileanBoostOperator":
        """The inverse of the operator.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector, CartesianDifferential3D
        >>> from galax.coordinates.operators import GalileanBoostOperator

        >>> op = GalileanBoostOperator(Quantity([1, 2, 3], "m/s"))
        >>> op.inverse
        GalileanBoostOperator( velocity=CartesianDifferential3D( ... ) )

        >>> op.inverse.velocity.d_x
        Quantity['speed'](Array(-1., dtype=float64), unit='m / s')

        """
        return GalileanBoostOperator(-self.velocity)

    # -----------------------------------------------------

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanBoostOperator", q: Abstract3DVector, t: Quantity["time"], /
    ) -> tuple[Abstract3DVector, Quantity["time"]]:
        """Apply the boost to the coordinates.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector, CartesianDifferential3D
        >>> from galax.coordinates.operators import GalileanBoostOperator

        >>> op = GalileanBoostOperator(Quantity([1, 2, 3], "m/s"))

        >>> q = Cartesian3DVector.constructor(Quantity([0, 0, 0], "m"))
        >>> t = Quantity(1, "s")
        >>> newq, newt = op(q, t)
        >>> newt
        Quantity['time'](Array(1, dtype=int64, ...), unit='s')
        >>> newq.x
        Quantity['length'](Array(1., dtype=float64), unit='m')

        """
        return q + self.velocity * t, t

    @op_call_dispatch
    def __call__(
        self: "GalileanBoostOperator",
        psp: AbstractPhaseSpacePositionBase,
        t: Quantity["time"],
        /,
    ) -> tuple[AbstractPhaseSpacePositionBase, Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector, CartesianDifferential3D
        >>> from galax.coordinates.operators import GalileanBoostOperator

        >>> op = GalileanBoostOperator(Quantity([1, 1, 1], "kpc/Gyr"))

        >>> psp = PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                          p=Quantity([0, 0, 0], "kpc/Gyr"))

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
        p = psp.p.represent_as(CartesianDifferential3D, psp.q).represent_as(
            type(psp.p), q
        )
        # Reasseble and return
        return (replace(psp, q=q, p=p), t)


##############################################################################
# Rotations


vec_matmul = quaxify(jnp.vectorize(jnp.matmul, signature="(3,3),(3)->(3)"))


@final
class GalileanRotationOperator(AbstractGalileanOperator):
    r"""Operator for Galilean rotations.

    The coordinate transform is given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t, R \mathbf{x})

    where :math:`R` is the rotation matrix.  Note this is NOT time dependent.

    Parameters
    ----------
    rotation : Array[float, (3, 3)]
        The rotation matrix.

    Raises
    ------
    ValueError
        If the rotation matrix is not orthogonal.

    Notes
    -----
    The Galilean rotation is not a time-dependent transformation.  This is part
    of the inhomogeneous Galilean group, which is the group of transformations
    that leave the space-time interval invariant.

    Examples
    --------
    We start with the required imports:

    >>> import jax.numpy as jnp
    >>> from jax_quantity import Quantity
    >>> import galax.coordinates.operators as gco

    We can then create a rotation operator:

    >>> theta = jnp.pi / 4  # 45 degrees
    >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
    ...                   [jnp.sin(theta), jnp.cos(theta),  0],
    ...                   [0,              0,               1]])
    >>> op = gco.GalileanRotationOperator(Rz)
    >>> op
    GalileanRotationOperator(rotation=f64[3,3])

    Translation operators can be applied to a Quantity[float, (N, 3), "...]:

    >>> q = Quantity([1, 0, 0], "m")
    >>> t = Quantity(1, "s")
    >>> newq, newt = op(q, t)
    >>> newq
    Quantity[...](Array([0.70710678, 0.70710678, 0. ], dtype=float64), unit='m')

    The time is not affected by the rotation.

    >>> newt
    Quantity['time'](Array(1, dtype=int64, ...), unit='s')

    This also works for a batch of vectors:

    >>> q = Quantity([[1, 0, 0], [0, 1, 0]], "m")
    >>> t = Quantity(0, "s")

    >>> newq, newt = op(q, t)
    >>> newq
    Quantity['length'](Array([[ 0.70710678,  0.70710678,  0.        ],
                              [-0.70710678,  0.70710678,  0.        ]], dtype=float64),
                       unit='m')

    Translation operators can be applied to :class:`vector.Abstract3DVector`:

    >>> from vector import Cartesian3DVector
    >>> q = Cartesian3DVector.constructor(q)  # from the previous example
    >>> newq, newt = op(q, t)
    >>> newq.x
    Quantity['length'](Array([ 0.70710678, -0.70710678], dtype=float64), unit='m')
    >>> newq.norm()
    Quantity['length'](Array([1., 1.], dtype=float64), unit='m')

    Translation operators can be applied to
    :class:`galax.coordinates.PhaseSpacePosition`:

    >>> from galax.coordinates import PhaseSpaceTimePosition
    >>> psp = PhaseSpaceTimePosition(q=q, p=Quantity([2, 0, 0], "km/s"), t=t)
    >>> newpsp = op(psp)
    >>> newpsp
    PhaseSpaceTimePosition(
      q=Cartesian3DVector( ... ),
      p=CartesianDifferential3D( ... ),
      t=Quantity[PhysicalType('time')](value=f64[1], unit=Unit("s"))
    )
    >>> newpsp.q.x
    Quantity['length'](Array([ 0.70710678, -0.70710678], dtype=float64), unit='m')
    >>> newpsp.p.d_x
    Quantity['speed'](Array(1.41421356, dtype=float64), unit='km / s')
    """

    rotation: Shaped[Array, "3 3"] = eqx.field(
        converter=lambda x: (
            x.rotation
            if isinstance(x, GalileanRotationOperator)
            else converter_float_array(x)
        )
    )
    """The rotation vector."""

    def __check_init__(self) -> None:
        # Check that the rotation matrix is orthogonal.
        if not jnp.allclose(self.rotation @ self.rotation.T, jnp.eye(3)):
            msg = "The rotation matrix must be orthogonal."
            raise ValueError(msg)

    # -----------------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean rotation is an inertial-frame preserving transform.

        Examples
        --------
        >>> import array_api_jax_compat as xp
        >>> from jax_quantity import Quantity
        >>> from galax.coordinates.operators import GalileanRotationOperator

        >>> theta = Quantity(45, "deg")
        >>> Rz = xp.asarray([[xp.cos(theta), -xp.sin(theta), 0],
        ...                  [xp.sin(theta), xp.cos(theta),  0],
        ...                  [0,             0,              1]])
        >>> op = GalileanRotationOperator(Rz)
        >>> op.is_inertial
        True
        """
        return True

    @property
    def inverse(self) -> "GalileanRotationOperator":
        """The inverse of the operator.

        Examples
        --------
        >>> import array_api_jax_compat as xp
        >>> from jax_quantity import Quantity
        >>> from galax.coordinates.operators import GalileanRotationOperator

        >>> theta = Quantity(45, "deg")
        >>> Rz = xp.asarray([[xp.cos(theta), -xp.sin(theta), 0],
        ...                  [xp.sin(theta), xp.cos(theta),  0],
        ...                  [0,             0,              1]])
        >>> op = GalileanRotationOperator(Rz)
        >>> op.inverse
        GalileanRotationOperator(rotation=f64[3,3])

        >>> jnp.allclose(op.rotation, op.inverse.rotation.T)
        Array(True, dtype=bool)
        """
        return replace(self, rotation=self.rotation.T)

    # -----------------------------------------------------

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanRotationOperator",
        q: Shaped[Quantity["length"], "*batch 3"],
        t: Quantity["time"],
        /,
    ) -> tuple[Shaped[Quantity["length"], "*batch 3"], Quantity["time"]]:
        """Apply the boost to the coordinates.

        Examples
        --------
        >>> import array_api_jax_compat as xp
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector, CartesianDifferential3D
        >>> from galax.coordinates.operators import GalileanRotationOperator

        >>> theta = Quantity(45, "deg")
        >>> Rz = xp.asarray([[xp.cos(theta), -xp.sin(theta), 0],
        ...                  [xp.sin(theta), xp.cos(theta),  0],
        ...                  [0,             0,              1]])
        >>> op = GalileanRotationOperator(Rz)

        >>> q = Quantity([1, 0, 0], "m")
        >>> t = Quantity(1, "s")
        >>> newq, newt = op(q, t)
        >>> newq
        Quantity[...](Array([0.70710678, 0.70710678, 0. ], dtype=float64), unit='m')

        The time is not affected by the rotation.
        >>> newt
        Quantity['time'](Array(1, dtype=int64, ...), unit='s')

        """
        return vec_matmul(self.rotation, q), t

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanRotationOperator", q: Abstract3DVector, t: Quantity["time"], /
    ) -> tuple[Abstract3DVector, Quantity["time"]]:
        """Apply the boost to the coordinates.

        Examples
        --------
        >>> import array_api_jax_compat as xp
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector, CartesianDifferential3D
        >>> from galax.coordinates.operators import GalileanRotationOperator

        >>> theta = Quantity(45, "deg")
        >>> Rz = xp.asarray([[xp.cos(theta), -xp.sin(theta), 0],
        ...                  [xp.sin(theta), xp.cos(theta),  0],
        ...                  [0,             0,              1]])
        >>> op = GalileanRotationOperator(Rz)

        >>> q = Cartesian3DVector.constructor(Quantity([1, 0, 0], "m"))
        >>> t = Quantity(1, "s")
        >>> newq, newt = op(q, t)
        >>> newq.x
        Quantity['length'](Array(0.70710678, dtype=float64), unit='m')

        The time is not affected by the rotation.
        >>> newt
        Quantity['time'](Array(1, dtype=int64, ...), unit='s')
        """
        vec = convert(  # Array[float, (N, 3)]
            q.represent_as(Cartesian3DVector).to_units("consistent"), Quantity
        )
        rcart = Cartesian3DVector.constructor(vec_matmul(self.rotation, vec))
        return rcart.represent_as(type(q)), t

    @op_call_dispatch
    def __call__(
        self: "GalileanRotationOperator",
        psp: AbstractPhaseSpacePositionBase,
        t: Quantity["time"],
        /,
    ) -> tuple[AbstractPhaseSpacePositionBase, Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> import array_api_jax_compat as xp
        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector, CartesianDifferential3D
        >>> from galax.coordinates.operators import GalileanRotationOperator
        >>> from galax.coordinates import PhaseSpacePosition

        >>> theta = Quantity(45, "deg")
        >>> Rz = xp.asarray([[xp.cos(theta), -xp.sin(theta), 0],
        ...                  [xp.sin(theta), xp.cos(theta),  0],
        ...                  [0,             0,              1]])
        >>> op = GalileanRotationOperator(Rz)

        >>> psp = PhaseSpacePosition(q=Quantity([1, 0, 0], "m"),
        ...                          p=Quantity([1, 0, 0], "m/s"))

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
        Quantity['time'](Array(1, dtype=int64, ...), unit='s')
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


##############################################################################
# Full Galilean operator


@final
class GalileanOperator(AbstractCompositeOperator, AbstractGalileanOperator):
    r"""Operator for general Galilean transformations.

    In the transformed frame the coordinates are given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t+s, R \mathbf{x} + \mathbf{v} t + \mathbf{a})

    where :math:`R` is the rotation matrix, :math:`\mathbf{v}` is the boost
    velocity, :math:`\mathbf{a}` is the spatial translationl, and :math:`s` is
    the time translation. This is equivalent to a sequential operation of 1. a
    rotation, 2. a translation, 3. a boost.

    Parameters
    ----------
    translation : `galax.coordinates.operators.GalileanTranslationOperator`
        The spatial translation of the frame. See
        :class:`galax.coordinates.operators.GalileanTranslationOperator` for
        alternative inputs to construct this parameter.
    velocity : :class:`galax.coordinates.operators.GalileanBoostOperator`
        The boost to the frame. See
        :class:`galax.coordinates.operators.GalileanBoostOperator` for
        alternative inputs to construct this parameter.

    Examples
    --------
    We start with the required imports:

    >>> from jax_quantity import Quantity
    >>> import galax.coordinates.operators as gco

    We can then create a Galilean operator:

    >>> op = gco.GalileanOperator(
    ...     translation=Quantity([0., 2., 3., 4.], "kpc"),
    ...     velocity=Quantity([1., 2., 3.], "km/s"))
    >>> op
    GalileanOperator(
      rotation=GalileanRotationOperator(rotation=f64[3,3]),
      translation=GalileanTranslationOperator(
        translation=FourVector(
          t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("kpc s / km")),
          q=Cartesian3DVector( ... ) )
      ),
      velocity=GalileanBoostOperator( velocity=CartesianDifferential3D( ... ) )
    )

    Note that the translation is a
    :class:`galax.coordinates.operators.GalileanTranslationOperator` with a
    :class:`vector.FourVector` translation, and the velocity is a
    :class:`galax.coordinates.operators.GalileanBoostOperator` with a
    :class:`vector.CartesianDifferential3D` velocity. We can also construct them
    directly, which allows for other vector types.

    >>> from vector import SphericalVector, FourVector, CartesianDifferential3D
    >>> op = gco.GalileanOperator(
    ...     translation=gco.GalileanTranslationOperator(
    ...         FourVector(t=Quantity(2.5, "Gyr"),
    ...                    q=SphericalVector(r=Quantity(1, "kpc"),
    ...                                      theta=Quantity(90, "deg"),
    ...                                      phi=Quantity(0, "rad") ) ) ),
    ...     velocity=gco.GalileanBoostOperator(
    ...         CartesianDifferential3D(d_x=Quantity(1, "km/s"),
    ...                                 d_y=Quantity(2, "km/s"),
    ...                                 d_z=Quantity(3, "km/s")))
    ... )
    >>> op
    GalileanOperator(
      rotation=GalileanRotationOperator(rotation=f64[3,3]),
      translation=GalileanTranslationOperator(
        translation=FourVector(
          t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("Gyr")),
          q=SphericalVector( ... ) )
      ),
      velocity=GalileanBoostOperator( velocity=CartesianDifferential3D( ... ) )
      )
    )

    Galilean operators can be applied to :class:`vector.FourVector`:

    >>> w = FourVector.constructor(Quantity([0, 0, 0, 0], "kpc"))
    >>> new = op(w)
    >>> new
    FourVector(
      t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("kpc s / km")),
      q=Cartesian3DVector( ... )
    )
    >>> new.t.to("Gyr")
    Quantity['time'](Array(2.5, dtype=float64), unit='Gyr')
    >>> new.q.x
    Quantity['length'](Array(3.55678041, dtype=float64), unit='kpc')

    Also the Galilean operators can also be applied to
    :class:`vector.Abstract3DVector` and :class:`jax_quantity.Quantity`:

    >>> q = Cartesian3DVector.constructor(Quantity([0, 0, 0], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> newq, newt = op(q, t)
    >>> newq.x
    Quantity['length'](Array(3.55678041, dtype=float64), unit='kpc')
    >>> newt
    Quantity['time'](Array(2.5, dtype=float64), unit='Gyr')

    As another example we can apply the operator to a
    :class:`~galax.coordinates.PhaseSpacePosition`:

    >>> from galax.coordinates import PhaseSpacePosition
    >>> psp = PhaseSpacePosition(q=Quantity([0, 0, 0], "kpc"),
    ...                          p=Quantity([0, 0, 0], "kpc/Gyr"))
    >>> newpsp, newt = op(psp, Quantity(1, "Gyr"))
    >>> newpsp.q.x
    Quantity['length'](Array(4.57949258, dtype=float64), unit='kpc')

    >>> newt
    Quantity['time'](Array(3.5, dtype=float64), unit='Gyr')
    """

    rotation: GalileanRotationOperator = eqx.field(
        default=GalileanRotationOperator(xp.eye(3)),
        converter=GalileanRotationOperator,
    )
    """The in-frame spatial rotation."""

    translation: GalileanTranslationOperator = eqx.field(
        default=GalileanTranslationOperator(Quantity([0, 0, 0, 0], "kpc")),
        converter=lambda x: (
            x
            if isinstance(x, GalileanTranslationOperator)
            else GalileanTranslationOperator(x)
        ),
    )
    """The temporal + spatial translation.

    The translation vector [T, Q].  This parameters accetps either a
    :class:`galax.coordinates.operators.GalileanTranslationOperator` instance or
    any input that can be used to construct a :meth:`vector.FourVector`, using
    :meth:`vector.FourVector.constructor`. See :class:`vector.FourVector` for
    details.
    """

    velocity: GalileanBoostOperator = eqx.field(
        default=GalileanBoostOperator(Quantity([0, 0, 0], "km/s")),
        converter=lambda x: (
            x if isinstance(x, GalileanBoostOperator) else GalileanBoostOperator(x)
        ),
    )
    """The boost to the frame.

    This parameters accepts either a
    :class:`galax.coordinates.operators.GalileanBoostOperator` instance or any
    input that can be used to construct a
    :class:`vector.CartesianDifferential3D`, using
    :meth:`vector.CartesianDifferential3D.constructor`. See
    :class:`vector.CartesianDifferential3D` for details.
    """

    @property
    def operators(
        self,
    ) -> tuple[
        GalileanRotationOperator, GalileanTranslationOperator, GalileanBoostOperator
    ]:
        """Rotation -> translateion -> boost."""
        return (self.rotation, self.translation, self.velocity)

    @overload
    def __getitem__(self, key: int) -> AbstractOperator:
        ...

    @overload
    def __getitem__(self, key: slice) -> "Self":
        ...

    def __getitem__(self, key: int | slice) -> "AbstractOperator | Self":
        if isinstance(key, int):
            return self.operators[key]
        return OperatorSequence(self.operators[key])


#####################################################################
# Simplification


@simplify_op.register
def _simplify_op(
    op: GalileanSpatialTranslationOperator, /, **kwargs: Any
) -> AbstractOperator:
    """Simplify a spatial translation operator."""
    # Check if the translation is zero.
    if jnp.allclose(convert(op.translation, Quantity).value, xp.zeros((3,)), **kwargs):
        return IdentityOperator()
    return op


@simplify_op.register
def _simplify_op(op: GalileanTranslationOperator, /, **kwargs: Any) -> AbstractOperator:
    """Simplify a translation operator."""
    # Check if the translation is zero.
    if jnp.allclose(convert(op.translation, Quantity).value, xp.zeros((4,)), **kwargs):
        return IdentityOperator()
    # Check if the translation is purely spatial.
    if op.translation[0] == 0:
        return GalileanSpatialTranslationOperator(op.translation[1:])
    return op


@simplify_op.register
def _simplify_op(op: GalileanBoostOperator, /, **kwargs: Any) -> AbstractOperator:
    """Simplify a boost operator."""
    # Check if the velocity is zero.
    if jnp.allclose(convert(op.velocity, Quantity).value, xp.zeros((3,)), **kwargs):
        return IdentityOperator()
    return op


@simplify_op.register
def _simplify_op(op: GalileanRotationOperator, /, **kwargs: Any) -> AbstractOperator:
    if jnp.allclose(op.rotation, xp.eye(3), **kwargs):
        return IdentityOperator()
    return op


@simplify_op.register
def _simplify_op(op: GalileanOperator, /, **kwargs: Any) -> AbstractOperator:
    """Simplify a Galilean operator."""
    # Check if all the sub-operators can be simplified to the identity.
    if all(
        isinstance(simplify_op(x, **kwargs), IdentityOperator) for x in op.operators
    ):
        return IdentityOperator()

    return op

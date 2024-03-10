"""galax: Galactic Dynamix in Jax."""

__all__ = ["AbstractPhaseSpacePositionBase"]

from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import equinox as eqx
import jax
from jaxtyping import Shaped
from plum import convert, dispatch

import quaxed.array_api as xp
from coordinax import (
    Abstract3DVector,
    Abstract3DVectorDifferential,
    AbstractVectorBase,
    Cartesian3DVector,
    represent_as as vector_represent_as,
)
from jax_quantity import Quantity

from galax.typing import BatchQVec3, BatchVec6
from galax.units import unitsystem

if TYPE_CHECKING:
    from typing import Self


@runtime_checkable
class ComponentShapeTuple(Protocol):
    """Shape tuple for phase-space positions."""

    q: int
    p: int

    def __iter__(self) -> Iterator[Any]: ...


class AbstractPhaseSpacePositionBase(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract base class for all the types of phase-space positions.

    Parameters
    ----------
    q : :class:`~vector.Abstract3DVector`
        Positions.
    p : :class:`~vector.Abstract3DVectorDifferential`
        Conjugate momenta at positions ``q``.

    See Also
    --------
    :class:`~galax.coordinates.AbstractPhaseSpacePosition`
    :class:`~galax.coordinates.AbstractPhaseSpaceTimePosition`
    """

    q: eqx.AbstractVar[Abstract3DVector]
    """Positions."""

    p: eqx.AbstractVar[Abstract3DVectorDifferential]
    """Conjugate momenta at positions ``q``."""

    # ==========================================================================
    # Array properties

    @property
    @abstractmethod
    def _shape_tuple(self) -> tuple[tuple[int, ...], Any]:
        """Batch, component shape."""
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the position and velocity arrays.

        This is the shape of the batch, not including the component shape.

        Returns
        -------
        shape : tuple[int, ...]
            The shape of the batch.

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian3DVector, CartesianDifferential3D
        >>> from galax.coordinates import PhaseSpacePosition

        We can create a phase-space position:

        >>> q = Cartesian3DVector(x=Quantity(1, "kpc"), y=Quantity(2, "kpc"),
        ...                       z=Quantity(3, "kpc"))
        >>> p = CartesianDifferential3D(d_x=Quantity(4, "km/s"),
        ...                             d_y=Quantity(5, "km/s"),
        ...                             d_z=Quantity(6, "km/s"))
        >>> pos = PhaseSpacePosition(q=q, p=p)

        We can access the shape of the position and velocity arrays:

        >>> pos.shape
        ()

        For a batch of phase-space positions, the shape will be non-empty:

        >>> q = Cartesian3DVector(x=Quantity([1, 4], "kpc"), y=Quantity(2, "kpc"),
        ...                       z=Quantity(3, "kpc"))
        >>> pos = PhaseSpacePosition(q=q, p=p)
        >>> pos.shape
        (2,)
        """
        return self._shape_tuple[0]

    @property
    def ndim(self) -> int:
        """Number of dimensions, not including component shape."""
        return len(self.shape)

    def __len__(self) -> int:
        """Return the number of particles."""
        return self.shape[0]

    @abstractmethod
    def __getitem__(self, index: Any) -> "Self": ...

    # ==========================================================================
    # Further Array properties

    @property
    def full_shape(self) -> tuple[int, ...]:
        """The full shape: batch and components.

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian3DVector, CartesianDifferential3D
        >>> from galax.coordinates import PhaseSpacePosition

        We can create a phase-space position:

        >>> q = Cartesian3DVector(x=Quantity([1], "kpc"), y=Quantity(2, "kpc"),
        ...                       z=Quantity(3, "kpc"))
        >>> p = CartesianDifferential3D(d_x=Quantity(4, "km/s"),
        ...                             d_y=Quantity(5, "km/s"),
        ...                             d_z=Quantity(6, "km/s"))
        >>> pos = PhaseSpacePosition(q=q, p=p)
        >>> pos.full_shape
        (1, 6)
        """
        batch_shape, component_shapes = self._shape_tuple
        return (*batch_shape, sum(component_shapes))

    # ==========================================================================
    # Convenience methods

    def w(self, *, units: Any) -> BatchVec6:
        """Phase-space position as an Array[float, (*batch, Q + P)].

        This is the full phase-space position, not including the time (if a
        component).

        Parameters
        ----------
        units : `galax.units.UnitSystem`, optional keyword-only
            The unit system. :func:`~galax.units.unitsystem` is used to
            convert the input to a unit system.

        Returns
        -------
        w : Array[float, (*batch, Q + P)]
            The phase-space position as a 6-vector in Cartesian coordinates.
            This will have shape
            :attr:`AbstractPhaseSpacePositionBase.full_shape`.

        Examples
        --------
        Assuming the following imports:

        >>> from jax_quantity import Quantity
        >>> from galax.coordinates import PhaseSpacePosition

        We can create a phase-space position and convert it to a 6-vector:

        >>> psp = PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                          p=Quantity([4, 5, 6], "km/s"))
        >>> psp.w(units="galactic")
        Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64)
        """
        usys = unitsystem(units)
        batch, comps = self._shape_tuple
        cart = self.represent_as(Cartesian3DVector)
        q = xp.broadcast_to(convert(cart.q, Quantity), (*batch, comps.q))
        p = xp.broadcast_to(convert(cart.p, Quantity), (*batch, comps.p))
        return xp.concat((q.decompose(usys).value, p.decompose(usys).value), axis=-1)

    def represent_as(self, /, target: type[AbstractVectorBase]) -> "Self":
        """Return with the components transformed.

        Parameters
        ----------
        target : type[:class:`~vector.AbstractVectorBase`]

        Returns
        -------
        w : :class:`~galax.coordinates.AbstractPhaseSpacePositionBase`
            The phase-space position with the components transformed.

        Examples
        --------
        With the following imports:

        >>> from jax_quantity import Quantity
        >>> from galax.coordinates import PhaseSpacePosition

        We can create a phase-space position and convert it to a 6-vector:

        >>> psp = PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                          p=Quantity([4, 5, 6], "km/s"))
        >>> psp.w(units="galactic")
        Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64)

        We can also convert it to a different representation:

        >>> from coordinax import CylindricalVector
        >>> psp.represent_as(CylindricalVector)
        PhaseSpacePosition( q=CylindricalVector(...), p=CylindricalDifferential(...) )
        """
        return cast("Self", vector_represent_as(self, target))

    # ==========================================================================
    # Dynamical quantities

    # TODO: property?
    @partial(jax.jit)
    def kinetic_energy(self) -> Shaped[Quantity["specific energy"], "*batch"]:
        r"""Return the specific kinetic energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2

        Returns
        -------
        E : Array[float, (*batch,)]
            The kinetic energy.

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian3DVector, CartesianDifferential3D
        >>> from galax.coordinates import PhaseSpacePosition

        We can construct a phase-space position:

        >>> q = Cartesian3DVector(
        ...     x=Quantity(1, "kpc"),
        ...     y=Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "kpc"),
        ...     z=Quantity(2, "kpc"))
        >>> p = CartesianDifferential3D(
        ...     d_x=Quantity(0, "km/s"),
        ...     d_y=Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "km/s"),
        ...     d_z=Quantity(0, "km/s"))
        >>> w = PhaseSpacePosition(q, p)

        We can compute the kinetic energy:

        >>> w.kinetic_energy()
        Quantity['specific energy'](Array([[0.5, 2. , 4.5, 8. ], [0.5, 2. , 4.5, 8. ]],
                                    dtype=float64), unit='km2 / s2')
        """
        return 0.5 * self.p.norm(self.q) ** 2

    # TODO: property?
    @partial(jax.jit)
    def angular_momentum(self) -> BatchQVec3:
        r"""Compute the angular momentum.

        .. math::

            \boldsymbol{{L}} = \boldsymbol{{q}} \times \boldsymbol{{p}}

        See :ref:`shape-conventions` for more information about the shapes of
        input and output objects.

        Returns
        -------
        L : Array[float, (*batch,3)]
            Array of angular momentum vectors in Cartesian coordinates.

        Examples
        --------
        We assume the following imports

        >>> from jax_quantity import Quantity
        >>> from galax.coordinates import PhaseSpacePosition

        We can compute the angular momentum of a single object

        >>> pos = Quantity([1., 0, 0], "au")
        >>> vel = Quantity([0, 2., 0], "au/yr")
        >>> w = PhaseSpacePosition(pos, vel)
        >>> w.angular_momentum()
        Quantity['diffusivity'](Array([0., 0., 2.], dtype=float64), unit='AU2 / yr')
        """
        # TODO: keep as a vector.
        #       https://github.com/GalacticDynamics/vector/issues/27
        cart = self.represent_as(Cartesian3DVector)
        q = convert(cart.q, Quantity)
        p = convert(cart.p, Quantity)
        return xp.linalg.cross(q, p)


# =============================================================================
# helper functions


@dispatch  # type: ignore[misc]
def represent_as(
    current: AbstractPhaseSpacePositionBase, target: type[AbstractVectorBase]
) -> AbstractPhaseSpacePositionBase:
    """Return with the components transformed."""
    return replace(
        current,
        q=current.q.represent_as(target),
        p=current.p.represent_as(target.differential_cls, current.q),
    )

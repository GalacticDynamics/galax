"""ABC for phase-space positions."""

__all__ = ["AbstractBasePhaseSpacePosition", "ComponentShapeTuple"]

from abc import abstractmethod
from collections.abc import Mapping
from functools import partial
from textwrap import indent
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import equinox as eqx
import jax
from jaxtyping import Shaped
from plum import convert, dispatch

import coordinax as cx
import quaxed.array_api as xp
from unxt import Quantity, unitsystem

import galax.typing as gt
from galax.utils.dataclasses import dataclass_items

if TYPE_CHECKING:
    from typing import Self


class ComponentShapeTuple(NamedTuple):
    """Component shape of the phase-space position."""

    q: int
    """Shape of the position component."""

    p: int
    """Shape of the momentum component."""

    t: int
    """Shape of the time component."""


class AbstractBasePhaseSpacePosition(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    r"""ABC underlying phase-space positions and their composites.

    The phase-space position is a point in the 3+3+1-dimensional phase space
    :math:`\mathbb{R}^7` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}\in\mathbb{R}^3`, the conjugate momentum
    :math:`\boldsymbol{p}\in\mathbb{R}^3`, and the time
    :math:`t\in\mathbb{R}^1`.
    """

    q: eqx.AbstractVar[cx.AbstractPosition3D]
    """Positions."""

    p: eqx.AbstractVar[cx.AbstractVelocity3D]
    """Conjugate momenta at positions ``q``."""

    t: eqx.AbstractVar[gt.BroadBatchFloatQScalar]
    """Time corresponding to the positions and momenta."""

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch  # type: ignore[misc]
    def constructor(
        cls: "type[AbstractBasePhaseSpacePosition]", obj: Mapping[str, Any], /
    ) -> "AbstractBasePhaseSpacePosition":
        """Construct from a mapping.

        Parameters
        ----------
        cls : type[:class:`~galax.coordinates.AbstractBasePhaseSpacePosition`]
            The class to construct.
        obj : Mapping[str, Any]
            The mapping from which to construct.

        Returns
        -------
        :class:`~galax.coordinates.AbstractBasePhaseSpacePosition`
            The constructed phase-space position.

        Examples
        --------
        With the following imports:

        >>> from unxt import Quantity
        >>> from galax.coordinates import PhaseSpacePosition

        We can create a phase-space position from a mapping:

        >>> obj = {"q": Quantity([1, 2, 3], "kpc"),
        ...        "p": Quantity([4, 5, 6], "km/s"),
        ...        "t": Quantity(0, "Gyr")}
        >>> PhaseSpacePosition.constructor(obj)
        PhaseSpacePosition(
            q=CartesianPosition3D( ... ),
            p=CartesianVelocity3D( ... ),
            t=Quantity[...](value=f64[], unit=Unit("Gyr"))
        )

        """
        return cls(**obj)

    # ==========================================================================

    def __str__(self) -> str:
        """Return a string representation of the object.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import galax.coordinates as gc
        >>> w = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                           p=Quantity([4, 5, 6], "km/s"),
        ...                           t=Quantity(-1, "Gyr"))
        >>> print(w)
        PhaseSpacePosition(
            q=<CartesianPosition3D (x[kpc], y[kpc], z[kpc])
                [1. 2. 3.]>,
            p=<CartesianVelocity3D (d_x[km / s], d_y[km / s], d_z[km / s])
                [4. 5. 6.]>,
            t=Quantity['time'](Array(-1., dtype=float64), unit='Gyr'))
        """
        fs = [indent(f"{k}={v!s}", "    ") for k, v in dataclass_items(self)]
        sep = ",\n" if len(fs) > 1 else ", "
        return f"{self.__class__.__name__}(\n{sep.join(fs)})"

    # ==========================================================================
    # Array properties

    @property
    @abstractmethod
    def _shape_tuple(self) -> tuple[gt.Shape, ComponentShapeTuple]:
        """Batch, component shape."""
        raise NotImplementedError

    @property
    def shape(self) -> gt.Shape:
        """Shape of the position and velocity arrays.

        This is the shape of the batch, not including the component shape.

        Returns
        -------
        shape : tuple[int, ...]
            The shape of the batch.

        Examples
        --------
        We assume the following imports:

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D, CartesianVelocity3D
        >>> from galax.coordinates import PhaseSpacePosition

        We can create a phase-space position:

        >>> q = CartesianPosition3D(x=Quantity(1, "kpc"), y=Quantity(2, "kpc"),
        ...                       z=Quantity(3, "kpc"))
        >>> p = CartesianVelocity3D(d_x=Quantity(4, "km/s"),
        ...                             d_y=Quantity(5, "km/s"),
        ...                             d_z=Quantity(6, "km/s"))
        >>> t = Quantity(0, "Gyr")
        >>> pos = PhaseSpacePosition(q=q, p=p, t=t)

        We can access the shape of the position and velocity arrays:

        >>> pos.shape
        ()

        For a batch of phase-space positions, the shape will be non-empty:

        >>> q = CartesianPosition3D(x=Quantity([1, 4], "kpc"), y=Quantity(2, "kpc"),
        ...                       z=Quantity(3, "kpc"))
        >>> pos = PhaseSpacePosition(q=q, p=p, t=t)
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
    def __getitem__(self, index: Any) -> "Self":
        """Return a new object with the given slice applied."""
        ...

    # ==========================================================================
    # Further Array properties

    @property
    def full_shape(self) -> gt.Shape:
        """The full shape: batch and components.

        Examples
        --------
        We assume the following imports:

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D, CartesianVelocity3D
        >>> from galax.coordinates import PhaseSpacePosition

        We can create a phase-space position:

        >>> q = CartesianPosition3D(x=Quantity([1], "kpc"), y=Quantity(2, "kpc"),
        ...                       z=Quantity(3, "kpc"))
        >>> p = CartesianVelocity3D(d_x=Quantity(4, "km/s"),
        ...                             d_y=Quantity(5, "km/s"),
        ...                             d_z=Quantity(6, "km/s"))
        >>> t = Quantity(0, "Gyr")
        >>> pos = PhaseSpacePosition(q=q, p=p, t=t)
        >>> pos.full_shape
        (1, 7)
        """
        batch_shape, component_shapes = self._shape_tuple
        return (*batch_shape, sum(component_shapes))

    # ==========================================================================
    # Convenience methods

    def w(self, *, units: Any) -> gt.BatchVec6:
        """Phase-space position as an Array[float, (*batch, Q + P)].

        This is the full phase-space position, not including the time (if a
        component).

        Parameters
        ----------
        units : `unxt.AbstractUnitSystem`, optional keyword-only
            The unit system. :func:`~unxt.unitsystem` is used to
            convert the input to a unit system.

        Returns
        -------
        w : Array[float, (*batch, Q + P)]
            The phase-space position as a 6-vector in Cartesian coordinates.
            This will have shape
            :attr:`AbstractPhaseSpacePosition.full_shape`.

        Examples
        --------
        Assuming the following imports:

        >>> from unxt import Quantity
        >>> from galax.coordinates import PhaseSpacePosition

        We can create a phase-space position and convert it to a 6-vector:

        >>> psp = PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                          p=Quantity([4, 5, 6], "km/s"),
        ...                          t=Quantity(0, "Gyr"))
        >>> psp.w(units="galactic")
        Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64)
        """
        usys = unitsystem(units)
        batch, comps = self._shape_tuple
        cart = self.represent_as(cx.CartesianPosition3D).to_units(usys)
        q = xp.broadcast_to(convert(cart.q, Quantity), (*batch, comps.q))
        p = xp.broadcast_to(convert(cart.p, Quantity), (*batch, comps.p))
        return xp.concat((q.value, p.value), axis=-1)

    def wt(self, *, units: Any) -> gt.BatchVec7:
        """Phase-space position as an Array[float, (*batch, 1+Q+P)].

        This is the full phase-space position, including the time.

        Parameters
        ----------
        units : `unxt.AbstractUnitSystem`, keyword-only
            The unit system. :func:`~unxt.unitsystem` is used to
            convert the input to a unit system.

        Returns
        -------
        wt : Array[float, (*batch, 1+Q+P)]
            The full phase-space position, including time on the first axis.

        Examples
        --------
        We assume the following imports:

        >>> from unxt import Quantity
        >>> from galax.coordinates import PhaseSpacePosition

        We can create a phase-space position and convert it to a 6-vector:

        >>> psp = PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                          p=Quantity([4, 5, 6], "km/s"),
        ...                          t=Quantity(7.0, "Myr"))
        >>> psp.wt(units="galactic")
            Array([7.00000000e+00, 1.00000000e+00, 2.00000000e+00, 3.00000000e+00,
                4.09084866e-03, 5.11356083e-03, 6.13627299e-03], dtype=float64)
        """
        usys = unitsystem(units)
        batch, comps = self._shape_tuple
        cart = self.represent_as(cx.CartesianPosition3D).to_units(usys)
        q = xp.broadcast_to(convert(cart.q, Quantity), (*batch, comps.q))
        p = xp.broadcast_to(convert(cart.p, Quantity), (*batch, comps.p))
        t = xp.broadcast_to(self.t.value[..., None], (*batch, comps.t))
        return xp.concat((t, q.value, p.value), axis=-1)

    def represent_as(
        self,
        position_cls: type[cx.AbstractPosition],
        /,
        differential_cls: type[cx.AbstractVelocity] | None = None,
    ) -> "Self":
        """Return with the components transformed.

        Parameters
        ----------
        position_cls : type[:class:`~vector.AbstractPosition`]
            The target position class.
        differential_cls : type[:class:`~vector.AbstractVelocity`], optional
            The target differential class. If `None` (default), the differential
            class of the target position class is used.

        Returns
        -------
        w : :class:`~galax.coordinates.AbstractPhaseSpacePosition`
            The phase-space position with the components transformed.

        Examples
        --------
        With the following imports:

        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from galax.coordinates import PhaseSpacePosition

        We can create a phase-space position and convert it to a 6-vector:

        >>> psp = PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                          p=Quantity([4, 5, 6], "km/s"),
        ...                          t=Quantity(0, "Gyr"))
        >>> psp.w(units="galactic")
        Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64)

        We can also convert it to a different representation:

        >>> psp.represent_as(cx.CylindricalPosition)
        PhaseSpacePosition( q=CylindricalPosition(...),
                            p=CylindricalVelocity(...),
                            t=Quantity[...](value=f64[], unit=Unit("Gyr")) )

        We can also convert it to a different representation with a different
        differential class:

        >>> psp.represent_as(cx.LonLatSphericalPosition, cx.LonCosLatSphericalVelocity)
        PhaseSpacePosition( q=LonLatSphericalPosition(...),
                            p=LonCosLatSphericalVelocity(...),
                            t=Quantity[...](value=f64[], unit=Unit("Gyr")) )
        """
        return cast("Self", cx.represent_as(self, position_cls, differential_cls))

    @abstractmethod
    def to_units(self, units: Any) -> "Self":
        """Return with the components transformed to the given unit system.

        Parameters
        ----------
        units : `unxt.AbstractUnitSystem`
            The unit system. :func:`~unxt.unitsystem` is used to
            convert the input to a unit system.

        Returns
        -------
        w : :class:`~galax.coordinates.AbstractPhaseSpacePosition`
            The phase-space position with the components transformed.

        Examples
        --------
        With the following imports:

        >>> from unxt import Quantity
        >>> from galax.coordinates import PhaseSpacePosition

        We can create a phase-space position and convert it to different units:

        >>> psp = PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                          p=Quantity([4, 5, 6], "km/s"),
        ...                          t=Quantity(0, "Gyr"))
        >>> psp.to_units("solarsystem")
        PhaseSpacePosition(
            q=CartesianPosition3D(
                x=Quantity[...](value=f64[], unit=Unit("AU")),
                ... ),
            p=CartesianVelocity3D(
                d_x=Quantity[...]( value=f64[], unit=Unit("AU / yr") ),
                ... ),
            t=Quantity[...](value=f64[], unit=Unit("yr"))
        )
        """
        ...

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

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D, CartesianVelocity3D
        >>> from galax.coordinates import PhaseSpacePosition

        We can construct a phase-space position:

        >>> q = CartesianPosition3D(
        ...     x=Quantity(1, "kpc"),
        ...     y=Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "kpc"),
        ...     z=Quantity(2, "kpc"))
        >>> p = CartesianVelocity3D(
        ...     d_x=Quantity(0, "km/s"),
        ...     d_y=Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "km/s"),
        ...     d_z=Quantity(0, "km/s"))
        >>> t = Quantity(0, "Gyr")
        >>> w = PhaseSpacePosition(q, p, t=t)

        We can compute the kinetic energy:

        >>> w.kinetic_energy()
        Quantity['specific energy'](Array([[0.5, 2. , 4.5, 8. ], [0.5, 2. , 4.5, 8. ]],
                                    dtype=float64), unit='km2 / s2')
        """
        return 0.5 * self.p.norm(self.q) ** 2

    def potential_energy(
        self, potential: "AbstractPotentialBase"
    ) -> Quantity["specific energy"]:
        r"""Return the specific potential energy.

        .. math::

            E_\Phi = \Phi(\boldsymbol{q})

        Parameters
        ----------
        potential : `galax.potential.AbstractPotentialBase`
            The potential object to compute the energy from.

        Returns
        -------
        E : Array[float, (*batch,)]
            The specific potential energy.

        Examples
        --------
        We assume the following imports:

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D, CartesianVelocity3D
        >>> from galax.coordinates import PhaseSpacePosition
        >>> from galax.potential import MilkyWayPotential

        We can construct a phase-space position:

        >>> q = CartesianPosition3D(
        ...     x=Quantity(1, "kpc"),
        ...     y=Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "kpc"),
        ...     z=Quantity(2, "kpc"))
        >>> p = CartesianVelocity3D(
        ...     d_x=Quantity(0, "km/s"),
        ...     d_y=Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "km/s"),
        ...     d_z=Quantity(0, "km/s"))
        >>> w = PhaseSpacePosition(q, p, t=Quantity(0, "Myr"))

        We can compute the kinetic energy:

        >>> pot = MilkyWayPotential()
        >>> w.potential_energy(pot)
        Quantity['specific energy'](Array(..., dtype=float64), unit='kpc2 / Myr2')
        """
        return potential.potential(self.q, t=self.t)

    @partial(jax.jit)
    def total_energy(self, potential: "AbstractPotentialBase") -> gt.BatchFloatQScalar:
        r"""Return the specific total energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2
            E_\Phi = \Phi(\boldsymbol{q})
            E = E_K + E_\Phi

        Parameters
        ----------
        potential : `galax.potential.AbstractPotentialBase`
            The potential object to compute the energy from.

        Returns
        -------
        E : Array[float, (*batch,)]
            The kinetic energy.

        Examples
        --------
        We assume the following imports:

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D, CartesianVelocity3D
        >>> from galax.coordinates import PhaseSpacePosition
        >>> from galax.potential import MilkyWayPotential

        We can construct a phase-space position:

        >>> q = CartesianPosition3D(
        ...     x=Quantity(1, "kpc"),
        ...     y=Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "kpc"),
        ...     z=Quantity(2, "kpc"))
        >>> p = CartesianVelocity3D(
        ...     d_x=Quantity(0, "km/s"),
        ...     d_y=Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "km/s"),
        ...     d_z=Quantity(0, "km/s"))
        >>> w = PhaseSpacePosition(q, p, t=Quantity(0, "Myr"))

        We can compute the kinetic energy:

        >>> pot = MilkyWayPotential()
        >>> w.total_energy(pot)
        Quantity['specific energy'](Array(..., dtype=float64), unit='km2 / s2')
        """
        return self.kinetic_energy() + self.potential_energy(potential)

    # TODO: property?
    @partial(jax.jit)
    def angular_momentum(self) -> gt.BatchQVec3:
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

        >>> from unxt import Quantity
        >>> from galax.coordinates import PhaseSpacePosition

        We can compute the angular momentum of a single object

        >>> q = Quantity([1., 0, 0], "au")
        >>> p = Quantity([0, 2., 0], "au/yr")
        >>> t = Quantity(0, "yr")
        >>> w = PhaseSpacePosition(q=q, p=p, t=t)
        >>> w.angular_momentum()
        Quantity[...](Array([0., 0., 2.], dtype=float64), unit='AU2 / yr')
        """
        # TODO: keep as a vector.
        #       https://github.com/GalacticDynamics/vector/issues/27
        cart = self.represent_as(cx.CartesianPosition3D)
        q = convert(cart.q, Quantity)
        p = convert(cart.p, Quantity)
        return xp.linalg.cross(q, p)


# =============================================================================
# helper functions

# -----------------------------------------------
# Register additional constructors


@AbstractBasePhaseSpacePosition.constructor._f.register  # type: ignore[misc]  # noqa: SLF001
def constructor(
    cls: type[AbstractBasePhaseSpacePosition], obj: AbstractBasePhaseSpacePosition, /
) -> AbstractBasePhaseSpacePosition:
    """Construct from a `AbstractBasePhaseSpacePosition`.

    Parameters
    ----------
    cls : type[:class:`~galax.coordinates.AbstractBasePhaseSpacePosition`]
        The class to construct.
    obj : :class:`~galax.coordinates.AbstractBasePhaseSpacePosition`
        The phase-space position object from which to construct.

    Returns
    -------
    :class:`~galax.coordinates.AbstractBasePhaseSpacePosition`
        The constructed phase-space position.

    Raises
    ------
    TypeError
        If the input object is not an instance of the target class.

    Examples
    --------
    With the following imports:

    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> from galax.coordinates import PhaseSpacePosition

    We can create a phase-space position and construct a new one from it:

    >>> psp = PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                          p=Quantity([4, 5, 6], "km/s"),
    ...                          t=Quantity(0, "Gyr"))
    >>> PhaseSpacePosition.constructor(psp) is psp
    True

    Note that the constructed object is the same as the input object because
    the types are the same. If we define a new class that inherits from
    :class:`~galax.coordinates.PhaseSpacePosition`, we can construct a
    new object from the input object that is an instance of the new class:

    >>> class NewPhaseSpacePosition(PhaseSpacePosition): pass
    >>> new_psp = NewPhaseSpacePosition.constructor(psp)
    >>> new_psp is psp
    False
    >>> isinstance(new_psp, NewPhaseSpacePosition)
    True

    """
    # TODO: add isinstance checks

    # Avoid copying if the types are the same. Isinstance is not strict
    # enough, so we use type() instead.
    if type(obj) is cls:  # pylint: disable=unidiomatic-typecheck
        return obj

    return cls(**dict(dataclass_items(obj)))

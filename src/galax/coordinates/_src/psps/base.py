"""ABC for phase-space positions."""

__all__ = ["AbstractBasePhaseSpacePosition", "ComponentShapeTuple"]

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from textwrap import indent
from typing import Any, NamedTuple, Self, cast

import equinox as eqx
import jax
from jaxtyping import Shaped
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items
from unxt.quantity import UncheckedQuantity as FastQ

import galax.typing as gt


class ComponentShapeTuple(NamedTuple):
    """Component shape of the phase-space position."""

    q: int
    """Shape of the position component."""

    p: int
    """Shape of the momentum component."""

    t: int
    """Shape of the time component."""


# =============================================================================


class AbstractBasePhaseSpacePosition(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    r"""ABC underlying phase-space positions and their composites.

    The phase-space position is a point in the 3+3+1-dimensional phase space
    :math:`\mathbb{R}^7` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}\in\mathbb{R}^3`, the conjugate momentum
    :math:`\boldsymbol{p}\in\mathbb{R}^3`, and the time
    :math:`t\in\mathbb{R}^1`.
    """

    q: eqx.AbstractVar[cx.vecs.AbstractPos3D]
    """Positions."""

    p: eqx.AbstractVar[cx.vecs.AbstractVel3D]
    """Conjugate momenta at positions ``q``."""

    t: eqx.AbstractVar[gt.BatchableFloatQScalar]
    """Time corresponding to the positions and momenta."""

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch  # type: ignore[misc]
    def from_(
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

        >>> import unxt as u
        >>> import galax.coordinates as gc

        We can create a phase-space position from a mapping:

        >>> obj = {"q": u.Quantity([1, 2, 3], "kpc"),
        ...        "p": u.Quantity([4, 5, 6], "km/s"),
        ...        "t": u.Quantity(0, "Gyr")}
        >>> gc.PhaseSpacePosition.from_(obj)
        PhaseSpacePosition(
            q=CartesianPos3D( ... ),
            p=CartesianVel3D( ... ),
            t=Quantity[...](value=...i64[], unit=Unit("Gyr"))
        )

        """
        return cls(**obj)

    # ==========================================================================

    def __str__(self) -> str:
        """Return a string representation of the object.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc
        >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
        ...                           p=u.Quantity([4, 5, 6], "km/s"),
        ...                           t=u.Quantity(-1, "Gyr"))
        >>> print(w)
        PhaseSpacePosition(
            q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
                [1 2 3]>,
            p=<CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
                [4 5 6]>,
            t=Quantity['time'](Array(-1, dtype=int64, ...), unit='Gyr'))
        """
        fs = [indent(f"{k}={v!s}", "    ") for k, v in field_items(self)]
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
        We require the following imports:

        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc

        We can create a phase-space position:

        >>> q = cx.CartesianPos3D.from_(u.Quantity([1, 2, 3], "kpc"))
        >>> p = cx.CartesianVel3D.from_(u.Quantity([4, 5, 6], "km/s"))
        >>> t = u.Quantity(0, "Gyr")
        >>> pos = gc.PhaseSpacePosition(q=q, p=p, t=t)

        We can access the shape of the position and velocity arrays:

        >>> pos.shape
        ()

        For a batch of phase-space positions, the shape will be non-empty:

        >>> q = cx.CartesianPos3D(x=u.Quantity([1, 4], "kpc"),
        ...                       y=u.Quantity(2, "kpc"),
        ...                       z=u.Quantity(3, "kpc"))
        >>> pos = gc.PhaseSpacePosition(q=q, p=p, t=t)
        >>> pos.shape
        (2,)
        """
        return self._shape_tuple[0]

    @property
    def ndim(self) -> int:
        """Number of dimensions, not including component shape."""
        return len(self.shape)

    def __len__(self) -> int:
        """Return the length of the leading batch dimension.

        Examples
        --------
        We require the following imports:

        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc

        We can create a phase-space position:

        >>> q = cx.CartesianPos3D.from_(u.Quantity([1, 2, 3], "kpc"))
        >>> p = cx.CartesianVel3D.from_(u.Quantity([4, 5, 6], "km/s"))
        >>> t = u.Quantity(0, "Gyr")
        >>> pos = gc.PhaseSpacePosition(q=q, p=p, t=t)
        >>> len(pos)
        0

        For a batch of phase-space positions, the length will be non-zero:

        >>> q = cx.CartesianPos3D(x=u.Quantity([1, 4], "kpc"),
        ...                       y=u.Quantity(2, "kpc"),
        ...                       z=u.Quantity(3, "kpc"))
        >>> pos = gc.PhaseSpacePosition(q=q, p=p, t=t)
        >>> len(pos)
        2
        """
        # scalars shape 0, instead of raising an error
        return self.shape[0] if self.shape else 0

    @dispatch  # type: ignore[misc]
    def __getitem__(
        self: "AbstractBasePhaseSpacePosition", index: Any
    ) -> "AbstractBasePhaseSpacePosition":
        """Return a new object with the given slice applied.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc

        >>> w = gc.PhaseSpacePosition(q=u.Quantity([[1, 2, 3]], "kpc"),
        ...                           p=u.Quantity([[4, 5, 6]], "km/s"),
        ...                           t=u.Quantity([0], "Gyr"))

        >>> w[jnp.array(0)]
        PhaseSpacePosition(
            q=CartesianPos3D( ... ),
            p=CartesianVel3D( ... ),
            t=Quantity[...](value=i64[], unit=Unit("Gyr"))
        )

        >>> w[jnp.array([0])]
        PhaseSpacePosition(
            q=CartesianPos3D( ... ),
            p=CartesianVel3D( ... ),
            t=Quantity[...](value=i64[1], unit=Unit("Gyr"))
        )

        """
        # The base assumption is to apply the index to all array fields
        arrays, non_arrays = eqx.partition(self, eqx.is_array)
        indexed_arrays = jax.tree.map(lambda x: x[index], arrays)
        new: "Self" = eqx.combine(indexed_arrays, non_arrays)
        return new

    # ---------------------------------------------------------------

    @dispatch.abstract  # type: ignore[misc]
    def __add__(
        self: "AbstractBasePhaseSpacePosition", other: Any, /
    ) -> "AbstractBasePhaseSpacePosition":
        """Add to a phase-space positions."""
        raise NotImplementedError  # pragma: no cover

    # ==========================================================================
    # Further Array properties

    @property
    def full_shape(self) -> gt.Shape:
        """The full shape: batch and components.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc

        We can create a phase-space position:

        >>> pos = gc.PhaseSpacePosition(q=u.Quantity([[1, 2, 3]], "kpc"),
        ...                             p=u.Quantity([4, 5, 6], "km/s"),
        ...                             t=u.Quantity(0, "Gyr"))
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

        >>> import unxt as u
        >>> import galax.coordinates as gc

        We can create a phase-space position and convert it to a 6-vector:

        >>> psp = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
        ...                             p=u.Quantity([4, 5, 6], "km/s"),
        ...                             t=u.Quantity(0, "Gyr"))
        >>> psp.w(units="galactic")
        Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64, ...)
        """
        usys = u.unitsystem(units)
        batch, comps = self._shape_tuple
        cart = self.vconvert(cx.CartesianPos3D)
        q = jnp.broadcast_to(
            u.ustrip(usys["length"], convert(cart.q, FastQ)), (*batch, comps.q)
        )
        p = jnp.broadcast_to(
            u.ustrip(usys["speed"], convert(cart.p, FastQ)), (*batch, comps.p)
        )
        return jnp.concat((q, p), axis=-1)

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
        >>> import unxt as u
        >>> import galax.coordinates as gc

        We can create a phase-space position and convert it to a 6-vector:

        >>> psp = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
        ...                             p=u.Quantity([4, 5, 6], "km/s"),
        ...                             t=u.Quantity(7.0, "Myr"))
        >>> psp.wt(units="galactic")
            Array([7.00000000e+00, 1.00000000e+00, 2.00000000e+00, 3.00000000e+00,
                4.09084866e-03, 5.11356083e-03, 6.13627299e-03], dtype=float64, ...)
        """
        usys = u.unitsystem(units)
        batch, comps = self._shape_tuple
        cart = self.vconvert(cx.CartesianPos3D).to_units(usys)
        q = jnp.broadcast_to(convert(cart.q, FastQ), (*batch, comps.q))
        p = jnp.broadcast_to(convert(cart.p, FastQ), (*batch, comps.p))
        t = jnp.broadcast_to(self.t.value[..., None], (*batch, comps.t))
        return jnp.concat((t, q.value, p.value), axis=-1)

    @dispatch(precedence=-1)
    def vconvert(
        self, target: Any, *args: Any, **kwargs: Any
    ) -> "AbstractBasePhaseSpacePosition":
        return cx.vconvert(target, self, *args, **kwargs)

    @dispatch
    def vconvert(
        self,
        position_cls: type[cx.vecs.AbstractPos],
        velocity_cls: type[cx.vecs.AbstractVel] | None = None,
        /,
        **kwargs: Any,
    ) -> "AbstractBasePhaseSpacePosition":
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
        w : :class:`~galax.coordinates.AbstractPhaseSpacePosition`
            The phase-space position with the components transformed.

        Examples
        --------
        With the following imports:

        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc

        We can create a phase-space position and convert it to a 6-vector:

        >>> psp = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
        ...                             p=u.Quantity([4, 5, 6], "km/s"),
        ...                             t=u.Quantity(0, "Gyr"))
        >>> psp.w(units="galactic")
        Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64, ...)

        We can also convert it to a different representation:

        >>> psp.vconvert(cx.vecs.CylindricalPos)
        PhaseSpacePosition( q=CylindricalPos(...),
                            p=CylindricalVel(...),
                            t=Quantity[...](value=...i64[], unit=Unit("Gyr")) )

        We can also convert it to a different representation with a different
        differential class:

        >>> psp.vconvert(cx.vecs.LonLatSphericalPos, cx.vecs.LonCosLatSphericalVel)
        PhaseSpacePosition( q=LonLatSphericalPos(...),
                            p=LonCosLatSphericalVel(...),
                            t=Quantity[...](value=...i64[], unit=Unit("Gyr")) )
        """
        return cast(
            "Self", cx.vconvert({"q": position_cls, "p": velocity_cls}, self, **kwargs)
        )

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
        >>> import unxt as u
        >>> import galax.coordinates as gc

        We can create a phase-space position and convert it to different units:

        >>> psp = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
        ...                             p=u.Quantity([4, 5, 6], "km/s"),
        ...                             t=u.Quantity(0, "Gyr"))
        >>> psp.to_units("solarsystem")
        PhaseSpacePosition(
            q=CartesianPos3D(
                x=Quantity[...](value=...f64[], unit=Unit("AU")),
                ... ),
            p=CartesianVel3D(
                d_x=Quantity[...]( value=...f64[], unit=Unit("AU / yr") ),
                ... ),
            t=Quantity[...](value=...f64[], unit=Unit("yr"))
        )
        """
        ...

    # ==========================================================================
    # Dynamical quantities

    # TODO: property?
    @partial(jax.jit, inline=True)
    def kinetic_energy(self) -> Shaped[u.Quantity["specific energy"], "*batch"]:
        r"""Return the specific kinetic energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2

        Returns
        -------
        E : Array[float, (*batch,)]
            The kinetic energy.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc

        We can construct a phase-space position:

        >>> q = cx.CartesianPos3D(
        ...     x=u.Quantity(1, "kpc"),
        ...     y=u.Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "kpc"),
        ...     z=u.Quantity(2, "kpc"))
        >>> p = cx.CartesianVel3D(
        ...     d_x=u.Quantity(0, "km/s"),
        ...     d_y=u.Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "km/s"),
        ...     d_z=u.Quantity(0, "km/s"))
        >>> t = u.Quantity(0, "Gyr")
        >>> w = gc.PhaseSpacePosition(q, p, t=t)

        We can compute the kinetic energy:

        >>> w.kinetic_energy()
        Quantity[...](Array([[0.5, 2. , 4.5, 8. ], [0.5, 2. , 4.5, 8. ]],
                            dtype=float64), unit='km2 / s2')
        """
        return 0.5 * self.p.norm(self.q) ** 2

    def potential_energy(
        self, potential: "AbstractBasePotential"
    ) -> u.Quantity["specific energy"]:
        r"""Return the specific potential energy.

        .. math::

            E_\Phi = \Phi(\boldsymbol{q})

        Parameters
        ----------
        potential : `galax.potential.AbstractBasePotential`
            The potential object to compute the energy from.

        Returns
        -------
        E : Array[float, (*batch,)]
            The specific potential energy.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp

        We can construct a phase-space position:

        >>> q = cx.CartesianPos3D(
        ...     x=u.Quantity(1, "kpc"),
        ...     y=u.Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "kpc"),
        ...     z=u.Quantity(2, "kpc"))
        >>> p = cx.CartesianVel3D(
        ...     d_x=u.Quantity(0, "km/s"),
        ...     d_y=u.Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "km/s"),
        ...     d_z=u.Quantity(0, "km/s"))
        >>> w = gc.PhaseSpacePosition(q, p, t=u.Quantity(0, "Myr"))

        We can compute the kinetic energy:

        >>> pot = gp.MilkyWayPotential()
        >>> w.potential_energy(pot)
        Quantity[...](Array(..., dtype=float64), unit='kpc2 / Myr2')
        """
        return potential.potential(self.q, t=self.t)

    @partial(jax.jit, inline=True)
    def total_energy(self, potential: "AbstractBasePotential") -> gt.BatchFloatQScalar:
        r"""Return the specific total energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2
            E_\Phi = \Phi(\boldsymbol{q})
            E = E_K + E_\Phi

        Parameters
        ----------
        potential : `galax.potential.AbstractBasePotential`
            The potential object to compute the energy from.

        Returns
        -------
        E : Array[float, (*batch,)]
            The kinetic energy.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp

        We can construct a phase-space position:

        >>> q = cx.CartesianPos3D(
        ...     x=u.Quantity(1, "kpc"),
        ...     y=u.Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "kpc"),
        ...     z=u.Quantity(2, "kpc"))
        >>> p = cx.CartesianVel3D(
        ...     d_x=u.Quantity(0, "km/s"),
        ...     d_y=u.Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "km/s"),
        ...     d_z=u.Quantity(0, "km/s"))
        >>> w = gc.PhaseSpacePosition(q, p, t=u.Quantity(0, "Myr"))

        We can compute the kinetic energy:

        >>> pot = gp.MilkyWayPotential()
        >>> w.total_energy(pot)
        Quantity[...](Array(..., dtype=float64), unit='km2 / s2')
        """
        return self.kinetic_energy() + self.potential_energy(potential)

    @partial(jax.jit, inline=True)
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

        >>> import unxt as u
        >>> import galax.coordinates as gc

        We can compute the angular momentum of a single object

        >>> q = u.Quantity([1., 0, 0], "au")
        >>> p = u.Quantity([0, 2., 0], "au/yr")
        >>> t = u.Quantity(0, "yr")
        >>> w = gc.PhaseSpacePosition(q=q, p=p, t=t)
        >>> w.angular_momentum()
        Quantity[...](Array([0., 0., 2.], dtype=float64), unit='AU2 / yr')
        """
        from galax.dynamics import specific_angular_momentum

        return specific_angular_momentum(self)


# =============================================================================
# Helper functions

# -----------------------------------------------
# Register additional constructors


@AbstractBasePhaseSpacePosition.from_.dispatch  # type: ignore[misc]
def from_(
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

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    We can create a phase-space position and construct a new one from it:

    >>> psp = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))
    >>> gc.PhaseSpacePosition.from_(psp) is psp
    True

    Note that the constructed object is the same as the input object because
    the types are the same. If we define a new class that inherits from
    :class:`~galax.coordinates.PhaseSpacePosition`, we can construct a
    new object from the input object that is an instance of the new class:

    >>> class NewPhaseSpacePosition(gc.PhaseSpacePosition): pass
    >>> new_psp = NewPhaseSpacePosition.from_(psp)
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

    return cls(**dict(field_items(obj)))


# -----------------------------------------------
# Addition


@AbstractBasePhaseSpacePosition.__add__.dispatch  # type: ignore[misc]
def add(
    self: AbstractBasePhaseSpacePosition,
    other: AbstractBasePhaseSpacePosition,
    /,
) -> AbstractBasePhaseSpacePosition:
    """Add two phase-space positions.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc

    >>> w1 = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                            p=u.Quantity([4, 5, 6], "km/s"),
    ...                            t=u.Quantity(0, "Gyr"))
    >>> w2 = gc.PhaseSpacePosition(q=u.Quantity([-1, -2, -3], "kpc"),
    ...                            p=u.Quantity([-4, -5, -6], "km/s"),
    ...                            t=u.Quantity(0, "Gyr"))
    >>> w3 = w1 + w2
    >>> w3
    PhaseSpacePosition(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity[PhysicalType('time')](value=...i64[], unit=Unit("Gyr"))
    )

    >>> w3.q.x.value
    Array(0, dtype=int64)

    If the times are different, an error is raised:

    >>> from dataclassish import replace
    >>> w4 = replace(w2, t=u.Quantity(1, "Gyr"))
    >>> try: w1 + w4
    ... except ValueError as e: print(e)
    Cannot add phase-space positions with different times

    """
    if not isinstance(other, type(self)):
        msg = f"Cannot add {type(self)} and {type(other)}"
        raise TypeError(msg)

    # Check the times are the same
    if not jnp.all(self.t == other.t):
        msg = "Cannot add phase-space positions with different times"
        raise ValueError(msg)

    # Add the fields
    return replace(self, q=self.q + other.q, p=self.p + other.p)

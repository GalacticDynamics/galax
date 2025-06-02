"""ABC for phase-space positions."""

__all__ = ["AbstractPhaseSpaceObject"]

import functools as ft
from abc import abstractmethod
from textwrap import indent
from typing import TYPE_CHECKING, Any, Self, cast

import equinox as eqx
import equinox.internal as eqxi
import jax
from jaxtyping import Real, Shaped
from plum import conversion_method, convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items, replace
from unxt.quantity import AbstractQuantity, BareQuantity as FastQ

import galax._custom_types as gt
from .utils import SLICE_ALL, PSPVConvertOptions
from galax.coordinates._src.frames import SimulationFrame

if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


# TODO: make it strict=True
class AbstractPhaseSpaceObject(cx.frames.AbstractCoordinate):  # type: ignore[misc]
    r"""ABC underlying phase-space positions and their composites."""

    q: eqx.AbstractVar[cx.vecs.AbstractPos3D]
    """Positions."""

    p: eqx.AbstractVar[cx.vecs.AbstractVel3D]
    """Conjugate momenta at positions ``q``."""

    frame: eqx.AbstractVar[SimulationFrame]  # TODO: support frames
    """The reference frame of the phase-space position."""

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch
    def from_(
        cls: "type[AbstractPhaseSpaceObject]", *args: Any, **kwargs: Any
    ) -> "AbstractPhaseSpaceObject":
        """Construct from arguments, defaulting to AbstractVector constructor."""
        return cast(AbstractPhaseSpaceObject, super().from_(*args, **kwargs))

    # ==========================================================================
    # Array API

    @property
    @abstractmethod
    def _shape_tuple(self) -> tuple[gt.Shape, Any]:
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
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc

        We can create a phase-space coordinate:

        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> p = cx.CartesianVel3D.from_([4, 5, 6], "km/s")
        >>> t = u.Quantity(0, "Gyr")
        >>> wt = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)

        We can access the shape of the position and velocity arrays:

        >>> wt.shape
        ()

        For a batch of phase-space positions, the shape will be non-empty:

        >>> q = cx.CartesianPos3D(x=u.Quantity([1, 4], "kpc"),
        ...                       y=u.Quantity(2, "kpc"),
        ...                       z=u.Quantity(3, "kpc"))
        >>> wt = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
        >>> wt.shape
        (2,)

        We can do the same with a phase-space position (lacking time):

        >>> w = gc.PhaseSpacePosition(q=q, p=p)
        >>> w.shape
        (2,)

        """
        return self._shape_tuple[0]

    @property
    def ndim(self) -> int:
        """Number of dimensions, not including component shape.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc

        >>> wt = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
        ...     p=u.Quantity([4, 5, 6], "km/s"), t=u.Quantity(0, "Gyr"))

        We can access the shape of the position and velocity arrays:

        >>> wt.shape
        ()

        For a batch of phase-space positions, the shape will be non-empty:

        >>> q = cx.CartesianPos3D(x=u.Quantity([1, 4], "kpc"),
        ...                       y=u.Quantity(2, "kpc"),
        ...                       z=u.Quantity(3, "kpc"))
        >>> w = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
        >>> w.shape
        (2,)

        """
        return len(self.shape)

    def __len__(self) -> int:
        """Return the length of the leading batch dimension.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc

        We can create a phase-space coordinate:

        >>> q = cx.CartesianPos3D.from_(u.Quantity([1, 2, 3], "kpc"))
        >>> p = cx.CartesianVel3D.from_(u.Quantity([4, 5, 6], "km/s"))
        >>> t = u.Quantity(0, "Gyr")
        >>> wt = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
        >>> len(wt)
        0

        For a batch of coordinate, the length will be non-zero:

        >>> q = cx.CartesianPos3D(x=u.Quantity([1, 4], "kpc"),
        ...                       y=u.Quantity(2, "kpc"),
        ...                       z=u.Quantity(3, "kpc"))
        >>> wt = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
        >>> len(wt)
        2

        This is also true for phase-space positions (lacking time):

        >>> w = gc.PhaseSpacePosition(q=q, p=p)
        >>> len(w)
        2

        """
        # scalars shape 0, instead of raising an error
        return self.shape[0] if self.shape else 0

    _GETITEM_DYNAMIC_FILTER_SPEC: AbstractClassVar[tuple[bool, ...]]

    @dispatch
    def __getitem__(
        self: "AbstractPhaseSpaceObject", index: Any
    ) -> "AbstractPhaseSpaceObject":
        r"""Return a new object with the given slice applied.

        This is the base dispatch, to directly apply the index to all array
        fields using `jax.tree.map` and filtering on `equinox.is_array`.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc

        - `galax.coordinates.PhaseSpaceCoordinate`:

        >>> wt = gc.PhaseSpaceCoordinate(q=u.Quantity([[1, 2, 3]], "kpc"),
        ...                              p=u.Quantity([[4, 5, 6]], "km/s"),
        ...                              t=u.Quantity([0], "Gyr"))

        >>> wt[jnp.array(0)]
        PhaseSpaceCoordinate(
            q=CartesianPos3D( ... ),
            p=CartesianVel3D( ... ),
            t=Quantity(0, unit='Gyr'),
            frame=SimulationFrame()
        )

        >>> wt[jnp.array([0])]
        PhaseSpaceCoordinate(
            q=CartesianPos3D( ... ),
            p=CartesianVel3D( ... ),
            t=Quantity([0], unit='Gyr'),
            frame=SimulationFrame()
        )

        - `galax.coordinates.PhaseSpacePosition`:

        >>> w = gc.PhaseSpacePosition(q=u.Quantity([[1, 2, 3]], "kpc"),
        ...                           p=u.Quantity([[4, 5, 6]], "km/s"))

        >>> w[()] is w
        True

        >>> print(w[jnp.array(0)])
        PhaseSpacePosition(
            q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
                [1 2 3]>,
            p=<CartesianVel3D (x[km / s], y[km / s], z[km / s])
                [4 5 6]>,
            frame=SimulationFrame())

        >>> print(w[jnp.array([0])])
        PhaseSpacePosition(
            q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
                [[1 2 3]]>,
            p=<CartesianVel3D (x[km / s], y[km / s], z[km / s])
                [[4 5 6]]>,
            frame=SimulationFrame())

        Slicing with int:

        >>> q = u.Quantity([[[1, 2, 3], [4, 5, 6]]], "m")
        >>> p = u.Quantity([[[7, 8, 9], [10, 11, 12]]], "m/s")
        >>> w = gc.PhaseSpacePosition(q=q, p=p)
        >>> w.shape
        (1, 2)

        >>> w[()] is w
        True

        >>> print(w[0], w[0].shape, sep='\n')
        PhaseSpacePosition(
            q=<CartesianPos3D (x[m], y[m], z[m])
                [[1 2 3]
                [4 5 6]]>,
            p=<CartesianVel3D (x[m / s], y[m / s], z[m / s])
                [[ 7  8  9]
                [10 11 12]]>,
            frame=SimulationFrame())
        (2,)

        >>> print(w[0, 1])
        PhaseSpacePosition(
            q=<CartesianPos3D (x[m], y[m], z[m])
                [4 5 6]>,
            p=<CartesianVel3D (x[m / s], y[m / s], z[m / s])
                [10 11 12]>,
            frame=SimulationFrame())

        """
        # Fast path [()]
        if isinstance(index, tuple) and len(index) == 0:
            return self
        # Fast path [slice(None)]
        if isinstance(index, slice) and index == SLICE_ALL:
            return self

        # Flatten by one level and partition into dynamic and static
        # where dynamic is q, p, ... and static is frame, ...
        leaves, treedef = eqx.tree_flatten_one_level(self)
        leaf_types = tuple(type(x) for x in leaves if x is not None)
        is_leaf = lambda x: isinstance(x, leaf_types)
        dynamic, static = eqx.partition(
            leaves, list(self._GETITEM_DYNAMIC_FILTER_SPEC), is_leaf=is_leaf
        )
        # Apply the index to the dynamic part
        dynamic = eqxi.ω(dynamic)[index].ω
        # Re-combine the dynamic and static parts
        leaves = eqx.combine(dynamic, static, is_leaf=is_leaf)
        # Rebuild the object
        w = jax.tree.unflatten(treedef, leaves)

        # The base assumption is to try to apply the index to all array fields
        return cast("Self", w)

    # ==========================================================================
    # Python API

    def __str__(self) -> str:
        """Return a string representation of the object.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc
        >>> wt = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
        ...                              p=u.Quantity([4, 5, 6], "km/s"),
        ...                              t=u.Quantity(-1, "Gyr"))
        >>> print(wt)
        PhaseSpaceCoordinate(
            q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
                [1 2 3]>,
            p=<CartesianVel3D (x[km / s], y[km / s], z[km / s])
                [4 5 6]>,
            t=Quantity(-1, unit='Gyr'),
            frame=SimulationFrame())

        >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
        ...                           p=u.Quantity([4, 5, 6], "km/s"))
        >>> print(w)
        PhaseSpacePosition(
            q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
                [1 2 3]>,
            p=<CartesianVel3D (x[km / s], y[km / s], z[km / s])
                [4 5 6]>,
            frame=SimulationFrame())

        """
        fs = [indent(f"{k}={v!s}", "    ") for k, v in field_items(self)]
        sep = ",\n" if len(fs) > 1 else ", "
        return f"{self.__class__.__name__}(\n{sep.join(fs)})"

    # ==========================================================================
    # Further Array properties

    @property
    def full_shape(self) -> gt.Shape:
        """The full shape: batch and components.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc

        >>> wt = gc.PhaseSpaceCoordinate(q=u.Quantity([[1, 2, 3]], "kpc"),
        ...                              p=u.Quantity([4, 5, 6], "km/s"),
        ...                              t=u.Quantity(0, "Gyr"))
        >>> wt.full_shape
        (1, 7)

        >>> w = gc.PhaseSpacePosition(q=u.Quantity([[1, 2, 3]], "kpc"),
        ...                           p=u.Quantity([4, 5, 6], "km/s"))
        >>> w.full_shape
        (1, 6)
        """
        batch_shape, component_shapes = self._shape_tuple
        return (*batch_shape, sum(component_shapes))

    # ==========================================================================
    # Convenience methods

    def _qp(
        self, *, units: u.AbstractUnitSystem
    ) -> tuple[Real[AbstractQuantity, "*batch 3"], Real[AbstractQuantity, "*batch 3"]]:
        """Return q,p as broadcasted quantities in a unit system.

        Returns
        -------
        Quantity[number, (*batch, Q), 'length']
        Quantity[number, (*batch, P), 'speed']

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc

        We convert a phase-space position to a 2-elt 3-vector tuple:

        >>> wt = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
        ...                              p=u.Quantity([4, 5, 6], "km/s"),
        ...                              t=u.Quantity(0, "Gyr"))
        >>> wt._qp(units=u.unitsystem("galactic"))
        (BareQuantity(Array([1, 2, 3], dtype=int64), unit='kpc'),
         BareQuantity(Array([0.00409085, 0.00511356, 0.00613627],
                                 dtype=float64, ...), unit='kpc / Myr'))

        >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
        ...                           p=u.Quantity([4, 5, 6], "km/s"))
        >>> w._qp(units=u.unitsystem("galactic"))
        (BareQuantity(Array([1, 2, 3], dtype=int64), unit='kpc'),
         BareQuantity(Array([0.00409085, 0.00511356, 0.00613627],
                                 dtype=float64, ...), unit='kpc / Myr'))

        """
        batch, comps = self._shape_tuple
        cart = self.vconvert(cx.CartesianPos3D)
        q = jnp.broadcast_to(
            u.uconvert(units["length"], convert(cart.q, FastQ)), (*batch, comps.q)
        )
        p = jnp.broadcast_to(
            u.uconvert(units["speed"], convert(cart.p, FastQ)), (*batch, comps.p)
        )
        return (q, p)

    def w(self, *, units: Any) -> gt.BtSz6:
        """Phase-space position as an Array[float, (*batch, Q + P)].

        This is the full phase-space position, not including the time (if a
        component).

        Parameters
        ----------
        units : Any, optional keyword-only
            The unit system. :func:`~unxt.unitsystem` is used to convert the
            input to a unit system.

        Returns
        -------
        Array[float, (*batch, Q + P)]
            The phase-space position as a 6-vector in Cartesian coordinates.
            This will have shape :attr:`AbstractOnePhaseSpaceObject.full_shape`.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc

        We can create a phase-space position and convert it to a 6-vector:

        >>> wt = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
        ...                              p=u.Quantity([4, 5, 6], "km/s"),
        ...                              t=u.Quantity(0, "Gyr"))
        >>> wt.w(units="galactic")
        Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64, ...)

        >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
        ...                           p=u.Quantity([4, 5, 6], "km/s"))
        >>> w.w(units="galactic")
        Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64, ...)

        """
        usys = u.unitsystem(units)
        q, p = self._qp(units=usys)
        return jnp.concat((q.ustrip(usys["length"]), p.ustrip(usys["speed"])), axis=-1)

    # ==========================================================================
    # Dynamical quantities

    # TODO: property?
    @ft.partial(jax.jit, inline=True)
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

        We can construct a phase-space coordinate:

        >>> q = cx.CartesianPos3D(
        ...     x=u.Quantity(1, "kpc"),
        ...     y=u.Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "kpc"),
        ...     z=u.Quantity(2, "kpc"))
        >>> p = cx.CartesianVel3D(
        ...     x=u.Quantity(0, "km/s"),
        ...     y=u.Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "km/s"),
        ...     z=u.Quantity(0, "km/s"))
        >>> t = u.Quantity(0, "Gyr")
        >>> wt = gc.PhaseSpaceCoordinate(q, p, t=t)

        We can compute the kinetic energy:

        >>> wt.kinetic_energy()
        Quantity(Array([[0.5, 2. , 4.5, 8. ], [0.5, 2. , 4.5, 8. ]],
                            dtype=float64), unit='km2 / s2')

        Also with a phase-space position (lacking time):

        >>> w = gc.PhaseSpacePosition(q, p)
        >>> w.kinetic_energy()
        Quantity(Array([[0.5, 2. , 4.5, 8. ], [0.5, 2. , 4.5, 8. ]],
                            dtype=float64), unit='km2 / s2')

        """
        return 0.5 * self.p.norm(self.q) ** 2

    @ft.partial(jax.jit, inline=True)
    def angular_momentum(self) -> cx.vecs.Cartesian3D:
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

        >>> wt = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
        >>> h = wt.angular_momentum()
        >>> print(h)
        <Cartesian3D (x[AU2 / yr], y[AU2 / yr], z[AU2 / yr])
            [0. 0. 2.]>

        >>> w = gc.PhaseSpacePosition(q=q, p=p)
        >>> h = w.angular_momentum()
        >>> print(h)
        <Cartesian3D (x[AU2 / yr], y[AU2 / yr], z[AU2 / yr])
            [0. 0. 2.]>

        """
        from galax.dynamics import specific_angular_momentum

        return specific_angular_momentum(self)


#####################################################################

# =========================================================
# Constructors


@AbstractPhaseSpaceObject.from_.dispatch  # type: ignore[attr-defined,misc]
def from_(
    cls: type[AbstractPhaseSpaceObject], obj: AbstractPhaseSpaceObject, /
) -> AbstractPhaseSpaceObject:
    """Construct from a `AbstractPhaseSpaceObject`.

    Parameters
    ----------
    cls : type[:class:`~galax.coordinates.AbstractPhaseSpaceObject`]
        The class to construct.
    obj : :class:`~galax.coordinates.AbstractPhaseSpaceObject`
        The phase-space position object from which to construct.

    Returns
    -------
    :class:`~galax.coordinates.AbstractPhaseSpaceObject`
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

    >>> psp = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                               p=u.Quantity([4, 5, 6], "km/s"),
    ...                               t=u.Quantity(0, "Gyr"))
    >>> gc.PhaseSpaceCoordinate.from_(psp) is psp
    True

    Note that the constructed object is the same as the input object because
    the types are the same. If we define a new class that inherits from
    :class:`~galax.coordinates.PhaseSpaceCoordinate`, we can construct a
    new object from the input object that is an instance of the new class:

    >>> class NewPhaseSpaceCoordinate(gc.PhaseSpaceCoordinate): pass
    >>> new_psp = NewPhaseSpaceCoordinate.from_(psp)
    >>> new_psp is psp
    False
    >>> isinstance(new_psp, NewPhaseSpaceCoordinate)
    True

    """
    # TODO: add isinstance checks

    # Avoid copying if the types are the same. Isinstance is not strict
    # enough, so we use type() instead.
    if type(obj) is cls:  # pylint: disable=unidiomatic-typecheck
        return obj

    return cls(**dict(field_items(obj)))


# =========================================================
# `coordinax.vconvert`


@dispatch
def vconvert(
    target: PSPVConvertOptions, wt: AbstractPhaseSpaceObject, /, **kw: Any
) -> AbstractPhaseSpaceObject:
    """Convert the phase-space coordinate to a different representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv
    >>> import galax.coordinates as gc

    We can create a phase-space coordinate and convert it to a 6-vector:

    >>> wt = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                              p=u.Quantity([4, 5, 6], "km/s"),
    ...                              t=u.Quantity(0, "Gyr"))

    Converting it to a different representation and differential class:

    >>> cx.vconvert({"q": cxv.LonLatSphericalPos, "p": cxv.LonCosLatSphericalVel}, wt)
    PhaseSpaceCoordinate( q=LonLatSphericalPos(...),
                          p=LonCosLatSphericalVel(...),
                          t=Quantity(0, unit='Gyr'),
                          frame=SimulationFrame() )

    """
    q_cls = target["q"]
    p_cls = q_cls.time_derivative_cls if (mayp := target.get("p")) is None else mayp
    return replace(wt, q=wt.q.vconvert(q_cls, **kw), p=wt.p.vconvert(p_cls, wt.q, **kw))


@dispatch
def vconvert(
    target_position_cls: type[cx.vecs.AbstractPos],
    wt: AbstractPhaseSpaceObject,
    /,
    **kw: Any,
) -> AbstractPhaseSpaceObject:
    """Convert the phase-space coordinate to a different representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    We can create a phase-space coordinate:

    >>> psp = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))

    Converting it to a different representation:

    >>> cx.vconvert(cx.vecs.CylindricalPos, psp)
    PhaseSpaceCoordinate( q=CylindricalPos(...),
                          p=CylindricalVel(...),
                          t=Quantity(0, unit='Gyr'),
                          frame=SimulationFrame() )

    If the new representation requires keyword arguments, they can be passed
    through:

    >>> cx.vconvert(cx.vecs.ProlateSpheroidalPos, psp, Delta=u.Quantity(2.0, "kpc"))
    PhaseSpaceCoordinate( q=ProlateSpheroidalPos(...),
                        p=ProlateSpheroidalVel(...),
                        t=Quantity(0, unit='Gyr'),
                        frame=SimulationFrame() )

    """
    target = {"q": target_position_cls, "p": target_position_cls.time_derivative_cls}
    return vconvert(target, wt, **kw)


# =========================================================
# `plum.convert`


@conversion_method(type_from=AbstractPhaseSpaceObject, type_to=cx.Coordinate)  # type: ignore[arg-type,type-abstract]
def convert_psp_to_coordinax_coordinate(
    obj: AbstractPhaseSpaceObject, /
) -> cx.Coordinate:
    """Convert a phase-space position to a coordinax coordinate.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc

    We can create a phase-space position and convert it to a coordinax coordinate:

    >>> psp = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                               p=u.Quantity([4, 5, 6], "km/s"),
    ...                               t=u.Quantity(0, "Gyr"))
    >>> convert(psp, cx.Coordinate)
    Coordinate(
        data=Space({ 'length': FourVector( ... ), 'speed': CartesianVel3D( ... ) }),
        frame=SimulationFrame()
    )

    """
    return cx.Coordinate(data=obj.data, frame=obj.frame)

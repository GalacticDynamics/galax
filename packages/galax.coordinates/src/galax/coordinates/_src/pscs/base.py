"""ABC for phase-space positions."""

__all__ = ["AbstractPhaseSpaceCoordinate", "ComponentShapeTuple"]

from abc import abstractmethod

from typing import TYPE_CHECKING, Any, NamedTuple, cast, override

import equinox as eqx
import equinox.internal as eqxi
import jax
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import Quantity as FastQ

import galax.coordinates.custom_types as gt
from galax.coordinates._src.base import AbstractPhaseSpaceObject
from galax.coordinates._src.frames import SimulationFrame
from galax.coordinates._src.utils import SLICE_ALL, getitem

if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


class ComponentShapeTuple(NamedTuple):
    """Component shape of the phase-space position."""

    q: int
    """Shape of the position component."""

    p: int
    """Shape of the momentum component."""

    t: int
    """Shape of the time component."""


# =============================================================================


class AbstractPhaseSpaceCoordinate(AbstractPhaseSpaceObject):
    r"""ABC underlying phase-space positions and their composites.

    The phase-space position is a point in the 3+3+1-dimensional phase space
    :math:`\mathbb{R}^7` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}\in\mathbb{R}^3`, the conjugate momentum
    :math:`\boldsymbol{p}\in\mathbb{R}^3`, and the time
    :math:`t\in\mathbb{R}^1`.

    Examples
    --------
    With the following imports:

    >>> import unxt as u
    >>> import galax.coordinates as gc

    We can create a phase-space position from a mapping:

    >>> obj = {"q": u.Q([1, 2, 3], "kpc"),
    ...        "p": u.Q([4, 5, 6], "km/s"),
    ...        "t": u.Q(0, "Gyr")}
    >>> gc.PhaseSpaceCoordinate.from_(obj)
    PhaseSpaceCoordinate( q=CartesianPos3D(...), p=CartesianVel3D(...),
                          t=Q(0, 'Gyr'), frame=SimulationFrame() )


    """

    q: eqx.AbstractVar[cx.vecs.AbstractPos3D]
    """Positions."""

    p: eqx.AbstractVar[cx.vecs.AbstractVel3D]
    """Conjugate momenta at positions ``q``."""

    t: eqx.AbstractVar[gt.BBtFloatQuSz0]
    """Time corresponding to the positions and momenta."""

    frame: eqx.AbstractVar[SimulationFrame]  # TODO: support frames
    """The reference frame of the phase-space coordinate."""

    _GETITEM_TIME_FILTER_SPEC: AbstractClassVar[tuple[bool, ...]]

    # ==========================================================================
    # Coordinate API

    @classmethod
    def _dimensionality(cls) -> int:
        """Return the dimensionality of the phase-space position.

        Examples
        --------
        >>> import galax.coordinates as gc
        >>> gc.PhaseSpaceCoordinate._dimensionality()
        7

        """
        return 7  # TODO: should it be 7? Also make it a Final

    @override
    @property
    def data(self) -> cx.KinematicSpace:  # type: ignore[misc]
        """Return the data as a space.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc

        We can create a phase-space position:

        >>> pos = gc.PhaseSpaceCoordinate(q=u.Q([1, 2, 3], "kpc"),
        ...                               p=u.Q([4, 5, 6], "km/s"),
        ...                               t=u.Q(0, "Gyr"))
        >>> pos.data
        KinematicSpace({ 'length': FourVector( ... ), 'speed': CartesianVel3D( ... ) })

        """
        return cx.KinematicSpace(
            length=cx.vecs.FourVector(t=self.t, q=self.q), speed=self.p
        )

    # ==========================================================================
    # Array API

    @property
    @abstractmethod
    def _shape_tuple(self) -> tuple[gt.Shape, ComponentShapeTuple]:
        """Batch, component shape."""
        raise NotImplementedError

    # ==========================================================================
    # Convenience methods

    def wt(self, *, units: Any) -> gt.BBtSz7:
        """Phase-space position as an Array[float, (*batch, 1+Q+P)].

        This is the full phase-space position, including the time.

        Parameters
        ----------
        units : `unxt.AbstractUnitSystem`, keyword-only
            The unit system. :func:`~unxt.unitsystem` is used to
            convert the input to a unit system.

        Returns
        -------
        Array[float, (*batch, (T=1)+Q+P)]
            The full phase-space position, including time on the first axis.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc

        We can create a phase-space position and convert it to a 7-vector:

        >>> psp = gc.PhaseSpaceCoordinate(q=u.Q([1, 2, 3], "kpc"),
        ...                               p=u.Q([4, 5, 6], "km/s"),
        ...                               t=u.Q(7.0, "Myr"))
        >>> psp.wt(units="galactic")
            Array([7.00000000e+00, 1.00000000e+00, 2.00000000e+00, 3.00000000e+00,
                4.09084866e-03, 5.11356083e-03, 6.13627299e-03], dtype=float64, ...)

        """
        usys = u.unitsystem(units)
        batch, comps = self._shape_tuple
        cart = self.vconvert(cx.CartesianPos3D).uconvert(usys)
        q = jnp.broadcast_to(convert(cart.q, FastQ), (*batch, comps.q))
        p = jnp.broadcast_to(convert(cart.p, FastQ), (*batch, comps.p))
        t = jnp.broadcast_to(self.t.ustrip(usys["time"])[..., None], (*batch, comps.t))
        return jnp.concat((t, q.value, p.value), axis=-1)  # type: ignore[no-any-return]


#####################################################################
# Dispatches

# ===============================================================
# `__getitem__`


@dispatch
def _psc_getitem_time_index(_: AbstractPhaseSpaceCoordinate, index: Any, /) -> Any:
    """Return the time index slicer. Default is to return as-is."""
    return index


@getitem.dispatch
def getitem(
    self: AbstractPhaseSpaceCoordinate, index: Any, /
) -> AbstractPhaseSpaceCoordinate:
    """Slice a PhaseSpaceCoordinate.

    The coordinate is partitioned into ``t``, ``qp`` -- separating the time from
    the rest. The index is applied to ``qp``. A separate time index is made
    given the coordinate and original index, then applied to ``t``. The whole
    thing is re-combined.

    Examples
    --------
    >>> from dataclasses import replace
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> q = u.Q([[[1, 2, 3], [4, 5, 6]]], "m")
    >>> p = u.Q([[[7, 8, 9], [10, 11, 12]]], "m/s")
    >>> t = u.Q(0, "Gyr")

    ## PhaseSpaceCoordinate

    >>> w = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)

    - `tuple`:

    >>> w[()] is w
    True

    >>> w[0, 1].q.x, w[0, 1].t
    (Q(4, 'm'), Q(0, 'Gyr'))

    >>> w[0, 1].q.x, w[0, 1].t
    (Q(4, 'm'), Q(0, 'Gyr'))

    >>> w = replace(w, t=u.Q([0], "Myr"))
    >>> w[0, 1].q.x, w[0, 1].t
    (Q(4, 'm'), Q(0, 'Myr'))

    >>> w = replace(w, t=u.Q([[[0],[1]]], "Myr"))
    >>> w[0, :].t
    Q([[0],
       [1]], 'Myr')

    - `slice` | `int`:

    >>> w = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
    >>> w[0].shape
    (2,)
    >>> w[0].t
    Q(0, 'Gyr')

    >>> w = gc.PhaseSpaceCoordinate(q=u.Q([[1, 2, 3]], "m"),
    ...                             p=u.Q([[4, 5, 6]], "m/s"),
    ...                             t=u.Q([7], "s"))
    >>> w[0].q.shape
    ()
    >>> w[0].t
    Q(7, 's')

    >>> w = gc.PhaseSpaceCoordinate(q=u.Q([[[1, 2, 3], [1, 2, 3]]], "m"),
    ...                             p=u.Q([[[4, 5, 6], [4, 5, 6]]], "m/s"),
    ...                             t=u.Q([[7]], "s"))
    >>> w[0].q.shape
    (2,)
    >>> w[0].t
    Q([7], 's')

    ## Orbit:

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
    # Split dynamic into time and qp
    time, qp = eqx.partition(
        dynamic, list(self._GETITEM_TIME_FILTER_SPEC), is_leaf=is_leaf
    )
    # Apply the index to the position fields (not the time)
    # TODO: restructure the index so that broadcasting of components is
    # unnecessary. E.g. (q (2), p () )[0] doesn't error.
    # Apply the index to the dynamic part
    qp = eqxi.ω(qp)[index].ω
    # Make and apply the time index
    tindex = _psc_getitem_time_index(self, index)
    time = eqxi.ω(time)[tindex].ω
    # Re-combine the leaves
    leaves = eqx.combine(time, qp, static, is_leaf=is_leaf)
    # Rebuild the object
    w = jax.tree.unflatten(treedef, leaves)
    return cast("AbstractPhaseSpaceCoordinate", w)

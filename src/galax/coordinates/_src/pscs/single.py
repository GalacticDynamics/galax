"""galax: Galactic Dynamics in Jax."""

__all__ = ["PhaseSpaceCoordinate"]

from dataclasses import KW_ONLY
from functools import partial
from typing import Any, ClassVar, final
from typing_extensions import override

import equinox as eqx
from plum import dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless

import galax.typing as gt
from .base import AbstractPhaseSpaceCoordinate, ComponentShapeTuple
from .base_composite import AbstractCompositePhaseSpaceCoordinate
from .base_single import AbstractBasicPhaseSpaceCoordinate
from galax.coordinates._src.base import AbstractPhaseSpaceObject
from galax.coordinates._src.frames import SimulationFrame, simulation_frame
from galax.utils._shape import batched_shape, vector_batched_shape


@final
class PhaseSpaceCoordinate(AbstractBasicPhaseSpaceCoordinate):
    r"""Phase-Space Position with time.

    The phase-space position is a point in the 7-dimensional phase space
    :math:`\mathbb{R}^7` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}`, the time :math:`t`, and the conjugate momentum
    :math:`\boldsymbol{p}`.

    Parameters
    ----------
    q : :class:`~coordinax.AbstractPos3D`
        A 3-vector of the positions, allowing for batched inputs.  This
        parameter accepts any 3-vector, e.g.
        :class:`~coordinax.SphericalPos`, or any input that can be used to
        make a :class:`~coordinax.CartesianPos3D` via
        :meth:`coordinax.vector`.
    p : :class:`~coordinax.AbstractVel3D`
        A 3-vector of the conjugate specific momenta at positions ``q``,
        allowing for batched inputs.  This parameter accepts any 3-vector
        differential, e.g.  :class:`~coordinax.SphericalVelocity`, or any input
        that can be used to make a :class:`~coordinax.CartesianVel3D` via
        :meth:`coordinax.vector`.
    t : Quantity[float, (*batch,), 'time']
        The time corresponding to the positions.

    Notes
    -----
    The batch shape of `q`, `p`, and `t` are broadcast together.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    Note that both `q` and `p` have convenience converters, allowing them to
    accept a variety of inputs when constructing a
    :class:`~coordinax.CartesianPos3D` or
    :class:`~coordinax.CartesianVel3D`, respectively.  For example,

    >>> t = u.Quantity(7, "s")
    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "m"),
    ...                           p=u.Quantity([4, 5, 6], "m/s"),
    ...                           t=t)
    >>> w
    PhaseSpaceCoordinate(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array(7, dtype=int64, ...), unit='s'),
      frame=SimulationFrame()
    )

    This can be done more explicitly:

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> p = cx.CartesianVel3D.from_([4, 5, 6], "m/s")

    >>> w2 = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
    >>> w2 == w
    Array(True, dtype=bool)

    When using the explicit constructors, the inputs can be any
    `coordinax.AbstractPos3D` and `coordinax.AbstractVel3D` types:

    >>> q = cx.SphericalPos(r=u.Quantity(1, "m"), theta=u.Quantity(2, "deg"),
    ...                     phi=u.Quantity(3, "deg"))
    >>> w3 = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
    >>> isinstance(w3.q, cx.SphericalPos)
    True

    Of course a similar effect can be achieved by using the
    `coordinax.vconvert` function (or convenience method on the phase-space
    position):

    >>> w4 = w3.vconvert(cx.SphericalPos, cx.CartesianVel3D)
    >>> w4
    PhaseSpaceCoordinate(
      q=SphericalPos( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array(7, dtype=int64, ...), unit='s'),
      frame=SimulationFrame()
    )

    """

    q: cx.vecs.AbstractPos3D = eqx.field(converter=cx.vector)
    """Positions, e.g CartesianPos3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    p: cx.vecs.AbstractVel3D = eqx.field(converter=cx.vector)
    r"""Conjugate momenta, e.g. CartesianVel3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    t: gt.TimeBBtSz0 | gt.SzN | gt.TimeSz0 = eqx.field(
        converter=u.Quantity["time"].from_
    )
    """The time corresponding to the positions.

    This is a Quantity with the same batch shape as the positions and
    velocities.  If `t` is a scalar it will be broadcast to the same batch shape
    as `q` and `p`.
    """

    _: KW_ONLY

    frame: SimulationFrame = eqx.field(
        default=simulation_frame,
        converter=Unless(
            cx.frames.AbstractReferenceFrame, cx.frames.TransformedReferenceFrame.from_
        ),
    )
    """The reference frame of the phase-space position."""

    _GETITEM_DYNAMIC_FILTER_SPEC: ClassVar = (True, True, True, False)
    _GETITEM_TIME_FILTER_SPEC: ClassVar = (False, False, True, False)

    # ==========================================================================
    # Array properties

    @property
    @override
    def _shape_tuple(self) -> tuple[gt.Shape, ComponentShapeTuple]:
        """Batch, component shape."""
        qbatch, qshape = vector_batched_shape(self.q)
        pbatch, pshape = vector_batched_shape(self.p)
        tbatch, _ = batched_shape(self.t, expect_ndim=0)
        tshape = 1
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch, tbatch)
        return batch_shape, ComponentShapeTuple(q=qshape, p=pshape, t=tshape)


#####################################################################
# Dispatches

# ===============================================================
# Constructors


# TODO: generalize
@AbstractPhaseSpaceCoordinate.from_.dispatch(precedence=1)
@partial(eqx.filter_jit, inline=True)
def from_(
    cls: type[PhaseSpaceCoordinate], obj: AbstractCompositePhaseSpaceCoordinate, /
) -> PhaseSpaceCoordinate:
    """Return a new PhaseSpaceCoordinate from the given object.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc


    >>> wt1 = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                               p=u.Quantity([4, 5, 6], "km/s"),
    ...                               t=u.Quantity(7, "Myr"))
    >>> wt2 = gc.PhaseSpaceCoordinate(q=u.Quantity([10, 20, 30], "kpc"),
    ...                               p=u.Quantity([40, 50, 60], "km/s"),
    ...                               t=u.Quantity(7, "Myr"))

    >>> cwt = gc.CompositePhaseSpaceCoordinate(wt1=wt1, wt2=wt2)

    >>> gc.PhaseSpaceCoordinate.from_(cwt)
    PhaseSpaceCoordinate(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array([7, 7], dtype=int64, ...), unit='Myr'),
      frame=SimulationFrame()
    )

    """
    return cls(q=obj.q, p=obj.p, t=obj.t)


@AbstractPhaseSpaceObject.from_.dispatch
def from_(
    cls: type[PhaseSpaceCoordinate],
    data: cx.Space,
    frame: cx.frames.AbstractReferenceFrame,
    /,
) -> PhaseSpaceCoordinate:
    """Return a new PhaseSpaceCoordinate from the given data and frame.

    Examples
    --------
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> data = cx.Space(length=cx.FourVector.from_([0, 1, 2, 3], "kpc"),
    ...                 speed=cx.CartesianVel3D.from_([4, 5, 6], "km/s"))

    >>> gc.PhaseSpaceCoordinate.from_(data, gc.frames.simulation_frame)
    PhaseSpaceCoordinate(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array(0., dtype=float64, ...), unit='kpc s / km'),
      frame=SimulationFrame()
    )

    """
    q4 = data["length"]
    q4 = eqx.error_if(
        q4,
        not isinstance(q4, cx.vecs.FourVector),
        "`data['length']` must be a FourVector",
    )
    return cls(q=q4.q, p=data["speed"], t=q4.t, frame=frame)


# ===============================================================
# `__getitem__`


@dispatch
def _psc_getitem_time_index(w: PhaseSpaceCoordinate, index: tuple[Any, ...], /) -> Any:
    """Return the time index slicer. Default is to return as-is."""
    match w.t.ndim:
        case 0:  # `t` is a scalar, return as is
            tindex = Ellipsis
        case 1 if len(index) == w.ndim:  # apply last index
            tindex = index[-1]
        case _:  # apply indices as normal
            tindex = index

    return tindex


@dispatch
def _psc_getitem_time_index(w: PhaseSpaceCoordinate, index: slice | int, /) -> Any:
    """Return the time index slicer. Default is to return as-is."""
    # Handle the time index
    match w.t.ndim:
        case 0:  # `t` is a scalar, return as is
            tindex = Ellipsis
        case 1 if w.ndim > 1:  # t vec on batched q, p
            tindex = Ellipsis
        case _:  # apply index as normal
            tindex = index

    return tindex

"""galax: Galactic Dynamics in Jax."""

__all__ = ["PhaseSpacePosition"]

from dataclasses import KW_ONLY, replace
from functools import partial
from typing import Any, final
from typing_extensions import override

import equinox as eqx

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Optional, Unless
from unxt.quantity import AbstractQuantity

import galax.typing as gt
from .base import AbstractBasePhaseSpacePosition, ComponentShapeTuple
from .base_composite import AbstractCompositePhaseSpacePosition
from .base_psp import AbstractPhaseSpacePosition
from galax.utils._shape import batched_shape, vector_batched_shape


@final
class PhaseSpacePosition(AbstractPhaseSpacePosition):
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
    t : Quantity[float, (*batch,), 'time'] | None
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
    >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "m"),
    ...                           p=u.Quantity([4, 5, 6], "m/s"),
    ...                           t=t)
    >>> w
    PhaseSpacePosition(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array(7, dtype=int64, ...), unit='s'),
      frame=NoFrame()
    )

    This can be done more explicitly:

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> p = cx.CartesianVel3D.from_([4, 5, 6], "m/s")

    >>> w2 = gc.PhaseSpacePosition(q=q, p=p, t=t)
    >>> w2 == w
    Array(True, dtype=bool)

    When using the explicit constructors, the inputs can be any
    `coordinax.AbstractPos3D` and `coordinax.AbstractVel3D` types:

    >>> q = cx.SphericalPos(r=u.Quantity(1, "m"), theta=u.Quantity(2, "deg"),
    ...                     phi=u.Quantity(3, "deg"))
    >>> w3 = gc.PhaseSpacePosition(q=q, p=p, t=t)
    >>> isinstance(w3.q, cx.SphericalPos)
    True

    Of course a similar effect can be achieved by using the
    `coordinax.vconvert` function (or convenience method on the phase-space
    position):

    >>> w4 = w3.vconvert(cx.SphericalPos, cx.CartesianVel3D)
    >>> w4
    PhaseSpacePosition(
      q=SphericalPos( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array(7, dtype=int64, ...), unit='s'),
      frame=NoFrame()
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

    t: gt.TimeBatchableScalar | gt.VecN | gt.TimeScalar | None = eqx.field(
        default=None,
        converter=Optional(u.Quantity["time"].from_),
    )
    """The time corresponding to the positions.

    This is a Quantity with the same batch shape as the positions and
    velocities.  If `t` is a scalar it will be broadcast to the same batch shape
    as `q` and `p`.
    """

    _: KW_ONLY

    frame: cx.frames.NoFrame = eqx.field(
        default=cx.frames.NoFrame(),
        converter=Unless(
            cx.frames.AbstractReferenceFrame, cx.frames.TransformedReferenceFrame.from_
        ),
    )
    """The reference frame of the phase-space position."""

    # ==========================================================================
    # Array properties

    @property
    @override
    def _shape_tuple(self) -> tuple[gt.Shape, ComponentShapeTuple]:
        """Batch, component shape."""
        qbatch, qshape = vector_batched_shape(self.q)
        pbatch, pshape = vector_batched_shape(self.p)
        tbatch: gt.Shape
        if self.t is None:
            tbatch, tshape = (), 0
        else:
            tbatch, _ = batched_shape(self.t, expect_ndim=0)
            tshape = 1
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch, tbatch)
        return batch_shape, ComponentShapeTuple(q=qshape, p=pshape, t=tshape)

    # ---------------------------------------------------------------
    # Getitem

    @AbstractBasePhaseSpacePosition.__getitem__.dispatch
    def __getitem__(
        self: "PhaseSpacePosition", index: tuple[Any, ...]
    ) -> "PhaseSpacePosition":
        """Return a new object with the given tuple selection applied.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc

        >>> q = u.Quantity([[[1, 2, 3], [4, 5, 6]]], "m")
        >>> p = u.Quantity([[[7, 8, 9], [10, 11, 12]]], "m/s")

        >>> w = gc.PhaseSpacePosition(q=q, p=p, t=None)
        >>> w[()] is w
        True

        >>> w = gc.PhaseSpacePosition(q=q, p=p, t=None)
        >>> w[0, 1].q.x
        Quantity['length'](Array(4, dtype=int64), unit='m')
        >>> w[0, 1].t is None
        True

        >>> w = gc.PhaseSpacePosition(q=q, p=p, t=u.Quantity(0, "Myr"))
        >>> w[0, 1].q.x
        Quantity['length'](Array(4, dtype=int64), unit='m')
        >>> w[0, 1].t
        Quantity['time'](Array(0, dtype=int64, ...), unit='Myr')

        >>> w = gc.PhaseSpacePosition(q=q, p=p, t=u.Quantity([0], "Myr"))
        >>> w[0, 1].q.x
        Quantity['length'](Array(4, dtype=int64), unit='m')
        >>> w[0, 1].t
        Quantity['time'](Array(0, dtype=int64), unit='Myr')

        >>> w = gc.PhaseSpacePosition(q=q, p=p, t=u.Quantity([[[0],[1]]], "Myr"))
        >>> w[0, :].t
        Quantity['time'](Array([[0], [1]], dtype=int64), unit='Myr')

        """
        # Empty selection w[()] should return the same object
        if len(index) == 0:
            return self

        # If `t` is None, then we can't index it
        if self.t is None:
            return replace(self, q=self.q[index], p=self.p[index])

        # Handle the time index
        #  - If `t` is a vector, then
        match self.t.ndim:
            case 0:  # `t` is a scalar, return as is
                tindex = Ellipsis
            case 1 if len(index) == self.ndim:  # apply last index
                tindex = index[-1]
            case _:  # apply indices as normal
                tindex = index

        return replace(self, q=self.q[index], p=self.p[index], t=self.t[tindex])

    @AbstractBasePhaseSpacePosition.__getitem__.dispatch
    def __getitem__(
        self: "PhaseSpacePosition", index: slice | int
    ) -> "PhaseSpacePosition":
        """Return a new object with the given slice selection applied.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc

        >>> q = u.Quantity([[[1, 2, 3], [4, 5, 6]]], "m")
        >>> p = u.Quantity([[[7, 8, 9], [10, 11, 12]]], "m/s")

        >>> w = gc.PhaseSpacePosition(q=q, p=p, t=None)
        >>> w[0].q.x
        Quantity['length'](Array([1, 4], dtype=int64), unit='m')

        >>> w = gc.PhaseSpacePosition(q=q, p=p, t=u.Quantity(0, "Myr"))
        >>> w[0].shape
        (2,)
        >>> w[0].t
        Quantity['time'](Array(0, dtype=int64, ...), unit='Myr')

        >>> w = gc.PhaseSpacePosition(q=u.Quantity([[1, 2, 3]], "m"),
        ...                           p=u.Quantity([[4, 5, 6]], "m/s"),
        ...                           t=u.Quantity([7], "s"))
        >>> w[0].q.shape
        ()
        >>> w[0].t
        Quantity['time'](Array(7, dtype=int64), unit='s')

        >>> w = gc.PhaseSpacePosition(q=u.Quantity([[[1, 2, 3], [1, 2, 3]]], "m"),
        ...                           p=u.Quantity([[[4, 5, 6], [4, 5, 6]]], "m/s"),
        ...                           t=u.Quantity([[7]], "s"))
        >>> w[0].q.shape
        (2,)
        >>> w[0].t
        Quantity['time'](Array([7], dtype=int64), unit='s')

        """
        # If `t` is None, then we can't index it
        if self.t is None:
            return replace(self, q=self.q[index], p=self.p[index])

        # Handle the time index
        match self.t.ndim:
            case 0:  # `t` is a scalar, return as is
                tindex = Ellipsis
            case 1 if self.ndim > 1:  # t vec on batched q, p
                tindex = Ellipsis
            case _:  # apply index as normal
                tindex = index

        # TODO: have to broadcast q, p
        return replace(self, q=self.q[index], p=self.p[index], t=self.t[tindex])

    # ==========================================================================
    # Convenience methods

    def wt(self, *, units: Any) -> gt.BatchVec7:
        """Return the phase-space-time position as a 7-vector.

        Raises
        ------
        Exception
            If the time is not defined.
        """
        _ = eqx.error_if(
            self.t, self.t is None, "No time defined for phase-space position"
        )
        return super().wt(units=units)


# TODO: generalize
@AbstractBasePhaseSpacePosition.from_.dispatch(precedence=1)
@partial(eqx.filter_jit, inline=True)
def from_(
    cls: type[PhaseSpacePosition], obj: AbstractCompositePhaseSpacePosition, /
) -> PhaseSpacePosition:
    """Return a new PhaseSpacePosition from the given object.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc


    >>> psp1 = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                              p=u.Quantity([4, 5, 6], "km/s"),
    ...                              t=u.Quantity(7, "Myr"))
    >>> psp2 = gc.PhaseSpacePosition(q=u.Quantity([10, 20, 30], "kpc"),
    ...                              p=u.Quantity([40, 50, 60], "km/s"),
    ...                              t=u.Quantity(7, "Myr"))

    >>> c_psp = gc.CompositePhaseSpacePosition(psp1=psp1, psp2=psp2)

    >>> gc.PhaseSpacePosition.from_(c_psp)
    PhaseSpacePosition(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array([7, 7], dtype=int64, ...), unit='Myr'),
      frame=NoFrame()
    )

    """
    return cls(q=obj.q, p=obj.p, t=obj.t)


@AbstractBasePhaseSpacePosition.from_.dispatch
def from_(
    cls: type[PhaseSpacePosition],
    data: cx.Space,
    frame: cx.frames.AbstractReferenceFrame,
    /,
) -> PhaseSpacePosition:
    """Return a new PhaseSpacePosition from the given data and frame.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> data = u.Quantity([1, 2, 3, 4, 5, 6], "m")
    >>> frame = cx.frames.Galactic()

    >>> gc.PhaseSpacePosition.from_(data, frame)
    PhaseSpacePosition(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=None,
      frame=Galactic()
    )

    """
    t: AbstractQuantity | None = None

    # TODO: more thorough constructor
    q = data["length"]
    if isinstance(q, cx.vecs.FourVector):
        t = q.t
        q = q.q

    return cls(q=q, p=data["speed"], t=t, frame=frame)

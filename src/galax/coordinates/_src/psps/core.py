"""galax: Galactic Dynamics in Jax."""

__all__ = ["PhaseSpacePosition"]

from functools import partial
from typing import Any, final
from typing_extensions import override

import equinox as eqx

import coordinax as cx
import quaxed.numpy as jnp
from dataclassish.converters import Optional
from unxt import Quantity

import galax.typing as gt
from .base import ComponentShapeTuple
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
    q : :class:`~coordinax.AbstractPosition3D`
        A 3-vector of the positions, allowing for batched inputs.  This
        parameter accepts any 3-vector, e.g.  :class:`~coordinax.SphericalPosition`,
        or any input that can be used to make a
        :class:`~coordinax.CartesianPosition3D` via
        :meth:`coordinax.AbstractPosition3D.constructor`.
    p : :class:`~coordinax.AbstractVelocity3D`
        A 3-vector of the conjugate specific momenta at positions ``q``,
        allowing for batched inputs.  This parameter accepts any 3-vector
        differential, e.g.  :class:`~coordinax.SphericalVelocity`, or any input
        that can be used to make a :class:`~coordinax.CartesianVelocity3D` via
        :meth:`coordinax.CartesianVelocity3D.constructor`.
    t : Quantity[float, (*batch,), 'time'] | None
        The time corresponding to the positions.

    Notes
    -----
    The batch shape of `q`, `p`, and `t` are broadcast together.

    Examples
    --------
    We assume the following imports:

    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    We can create a phase-space position:

    >>> q = cx.CartesianPosition3D(x=Quantity(1, "m"), y=Quantity(2, "m"),
    ...                            z=Quantity(3, "m"))
    >>> p = cx.CartesianVelocity3D(d_x=Quantity(4, "m/s"), d_y=Quantity(5, "m/s"),
    ...                            d_z=Quantity(6, "m/s"))
    >>> t = Quantity(7.0, "s")

    >>> psp = gc.PhaseSpacePosition(q=q, p=p, t=t)
    >>> psp
    PhaseSpacePosition(
      q=CartesianPosition3D(
        x=Quantity[...](value=f64[], unit=Unit("m")),
        y=Quantity[...](value=f64[], unit=Unit("m")),
        z=Quantity[...](value=f64[], unit=Unit("m"))
      ),
      p=CartesianVelocity3D(
        d_x=Quantity[...]( value=f64[], unit=Unit("m / s") ),
        d_y=Quantity[...]( value=f64[], unit=Unit("m / s") ),
        d_z=Quantity[...]( value=f64[], unit=Unit("m / s") )
      ),
      t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("s"))
    )

    Note that both `q` and `p` have convenience converters, allowing them to
    accept a variety of inputs when constructing a
    :class:`~coordinax.CartesianPosition3D` or
    :class:`~coordinax.CartesianVelocity3D`, respectively.  For example,

    >>> w2 = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "m"),
    ...                            p=Quantity([4, 5, 6], "m/s"),
    ...                            t=t)
    >>> w2 == psp
    Array(True, dtype=bool)

    """

    q: cx.AbstractPosition3D = eqx.field(converter=cx.AbstractPosition3D.constructor)
    """Positions, e.g CartesianPosition3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    p: cx.AbstractVelocity3D = eqx.field(converter=cx.AbstractVelocity3D.constructor)
    r"""Conjugate momenta, e.g. CartesianVelocity3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    t: gt.TimeBatchableScalar | gt.TimeScalar | None = eqx.field(
        default=None,
        converter=Optional(partial(Quantity["time"].constructor, dtype=float)),
    )
    """The time corresponding to the positions.

    This is a Quantity with the same batch shape as the positions and
    velocities.  If `t` is a scalar it will be broadcast to the same batch shape
    as `q` and `p`.
    """

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

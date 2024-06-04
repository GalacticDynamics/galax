"""galax: Galactic Dynamics in Jax."""

__all__ = ["PhaseSpacePosition"]

from typing import Any, NamedTuple, final

import equinox as eqx
import jax.numpy as jnp
from typing_extensions import override

from coordinax import AbstractPosition3D, AbstractVelocity3D
from unxt import Quantity

import galax.typing as gt
from .base_psp import AbstractPhaseSpacePosition
from .utils import _p_converter, _q_converter
from galax.utils._shape import batched_shape, expand_batch_dims, vector_batched_shape


class ComponentShapeTuple(NamedTuple):
    """Component shape of the phase-space position."""

    q: int
    """Shape of the position."""

    p: int
    """Shape of the momentum."""

    t: int | None
    """Shape of the time."""


def _converter_t(x: Any) -> gt.BroadBatchFloatQScalar | gt.FloatQScalar | None:
    """Convert `t` to Quantity."""
    return Quantity["time"].constructor(x, dtype=float) if x is not None else None


@final
class PhaseSpacePosition(AbstractPhaseSpacePosition):
    r"""Phase-Space Position with time.

    The phase-space position is a point in the 7-dimensional phase space
    :math:`\mathbb{R}^7` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}`, the time :math:`t`, and the conjugate momentum
    :math:`\boldsymbol{p}`.

    Parameters
    ----------
    q : :class:`~vector.AbstractPosition3D`
        A 3-vector of the positions, allowing for batched inputs.  This
        parameter accepts any 3-vector, e.g.  :class:`~vector.SphericalPosition`,
        or any input that can be used to make a
        :class:`~vector.CartesianPosition3D` via
        :meth:`vector.AbstractPosition3D.constructor`.
    p : :class:`~vector.AbstractVelocity3D`
        A 3-vector of the conjugate specific momenta at positions ``q``,
        allowing for batched inputs.  This parameter accepts any 3-vector
        differential, e.g.  :class:`~vector.SphericalVelocity`, or any input
        that can be used to make a :class:`~vector.CartesianVelocity3D` via
        :meth:`vector.CartesianVelocity3D.constructor`.
    t : Quantity[float, (*batch,), 'time'] | None
        The time corresponding to the positions.

    Notes
    -----
    The batch shape of `q`, `p`, and `t` are broadcast together.

    Examples
    --------
    We assume the following imports:

    >>> from unxt import Quantity
    >>> from coordinax import CartesianPosition3D, CartesianVelocity3D
    >>> from galax.coordinates import PhaseSpacePosition

    We can create a phase-space position:

    >>> q = CartesianPosition3D(x=Quantity(1, "m"), y=Quantity(2, "m"),
    ...                       z=Quantity(3, "m"))
    >>> p = CartesianVelocity3D(d_x=Quantity(4, "m/s"), d_y=Quantity(5, "m/s"),
    ...                             d_z=Quantity(6, "m/s"))
    >>> t = Quantity(7.0, "s")

    >>> psp = PhaseSpacePosition(q=q, p=p, t=t)
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
    :class:`~vector.CartesianPosition3D` or
    :class:`~vector.CartesianVelocity3D`, respectively.  For example,

    >>> psp2 = PhaseSpacePosition(q=Quantity([1, 2, 3], "m"),
    ...                           p=Quantity([4, 5, 6], "m/s"), t=t)
    >>> psp2 == psp
    Array(True, dtype=bool)

    """

    q: AbstractPosition3D = eqx.field(converter=_q_converter)
    """Positions, e.g CartesianPosition3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    p: AbstractVelocity3D = eqx.field(converter=_p_converter)
    r"""Conjugate momenta, e.g. CartesianVelocity3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    t: gt.BroadBatchFloatQScalar | gt.FloatQScalar | None = eqx.field(
        default=None, converter=_converter_t
    )
    """The time corresponding to the positions.

    This is a Quantity with the same batch shape as the positions and
    velocities.  If `t` is a scalar it will be broadcast to the same batch shape
    as `q` and `p`.
    """

    def __post_init__(self) -> None:
        """Post-initialization."""
        # Need to ensure t shape is correct. Can be Vec0.
        if (t := self.t) is not None and t.ndim in (0, 1):
            t = expand_batch_dims(t, ndim=self.q.ndim - t.ndim)
            object.__setattr__(self, "t", t)

    # ==========================================================================
    # Array properties

    @property
    @override
    def _shape_tuple(self) -> tuple[gt.Shape, ComponentShapeTuple]:  # type: ignore[override]
        """Batch, component shape."""
        qbatch, qshape = vector_batched_shape(self.q)
        pbatch, pshape = vector_batched_shape(self.p)
        tbatch: gt.Shape
        if self.t is None:
            tbatch, tshape = (), None
        else:
            tbatch, _ = batched_shape(self.t, expect_ndim=0)
            tshape = 1
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch, tbatch)
        return batch_shape, ComponentShapeTuple(q=qshape, p=pshape, t=tshape)

    # ==========================================================================
    # Convenience methods

    def wt(self, *, units: Any) -> gt.BatchVec7:
        _ = eqx.error_if(
            self.t, self.t is None, "No time defined for phase-space position"
        )
        return super().wt(units=units)

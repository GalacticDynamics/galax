"""galax: Galactic Dynamix in Jax."""

__all__ = ["PhaseSpacePosition"]

from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
from plum import convert

import quaxed.array_api as xp
from coordinax import Abstract3DVector, Abstract3DVectorDifferential, Cartesian3DVector
from unxt import Quantity

from .base import AbstractPhaseSpacePosition, ComponentShapeTuple
from .utils import _p_converter, _q_converter
from galax.typing import BatchVec7, BroadBatchFloatQScalar, QVec1
from galax.units import unitsystem
from galax.utils._shape import batched_shape, expand_batch_dims, vector_batched_shape


@final
class PhaseSpacePosition(AbstractPhaseSpacePosition):
    r"""Phase-Space Position with time.

    The phase-space position is a point in the 7-dimensional phase space
    :math:`\\mathbb{R}^7` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}`, the time :math:`t`, and the conjugate momentum
    :math:`\boldsymbol{p}`.

    Parameters
    ----------
    q : :class:`~vector.Abstract3DVector`
        A 3-vector of the positions, allowing for batched inputs.  This
        parameter accepts any 3-vector, e.g.  :class:`~vector.SphericalVector`,
        or any input that can be used to make a
        :class:`~vector.Cartesian3DVector` via
        :meth:`vector.Abstract3DVector.constructor`.
    p : :class:`~vector.Abstract3DVectorDifferential`
        A 3-vector of the conjugate specific momenta at positions ``q``,
        allowing for batched inputs.  This parameter accepts any 3-vector
        differential, e.g.  :class:`~vector.SphericalDifferential`, or any input
        that can be used to make a :class:`~vector.CartesianDifferential3D` via
        :meth:`vector.CartesianDifferential3D.constructor`.
    t : Quantity[float, (*batch,), 'time']
        The time corresponding to the positions.

    Notes
    -----
    The batch shape of `q`, `p`, and `t` are broadcast together.

    Examples
    --------
    We assume the following imports:

    >>> from unxt import Quantity
    >>> from coordinax import Cartesian3DVector, CartesianDifferential3D
    >>> from galax.coordinates import PhaseSpacePosition

    We can create a phase-space position:

    >>> q = Cartesian3DVector(x=Quantity(1, "m"), y=Quantity(2, "m"),
    ...                       z=Quantity(3, "m"))
    >>> p = CartesianDifferential3D(d_x=Quantity(4, "m/s"), d_y=Quantity(5, "m/s"),
    ...                             d_z=Quantity(6, "m/s"))
    >>> t = Quantity(7.0, "s")

    >>> psp = PhaseSpacePosition(q=q, p=p, t=t)
    >>> psp
    PhaseSpacePosition(
      q=Cartesian3DVector(
        x=Quantity[...](value=f64[], unit=Unit("m")),
        y=Quantity[...](value=f64[], unit=Unit("m")),
        z=Quantity[...](value=f64[], unit=Unit("m"))
      ),
      p=CartesianDifferential3D(
        d_x=Quantity[...]( value=f64[], unit=Unit("m / s") ),
        d_y=Quantity[...]( value=f64[], unit=Unit("m / s") ),
        d_z=Quantity[...]( value=f64[], unit=Unit("m / s") )
      ),
      t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("s"))
    )

    Note that both `q` and `p` have convenience converters, allowing them to
    accept a variety of inputs when constructing a
    :class:`~vector.Cartesian3DVector` or
    :class:`~vector.CartesianDifferential3D`, respectively.  For example,

    >>> psp2 = PhaseSpacePosition(q=Quantity([1, 2, 3], "m"),
    ...                               p=Quantity([4, 5, 6], "m/s"), t=t)
    >>> psp2 == psp
    Array(True, dtype=bool)

    """

    q: Abstract3DVector = eqx.field(converter=_q_converter)
    """Positions, e.g Cartesian3DVector.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    p: Abstract3DVectorDifferential = eqx.field(converter=_p_converter)
    r"""Conjugate momenta, e.g. CartesianDifferential3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    t: BroadBatchFloatQScalar | QVec1 = eqx.field(
        converter=Quantity["time"].constructor
    )
    """The time corresponding to the positions.

    This is a Quantity with the same batch shape as the positions and
    velocities.  If `t` is a scalar it will be broadcast to the same batch shape
    as `q` and `p`.
    """

    def __post_init__(self) -> None:
        """Post-initialization."""
        # Need to ensure t shape is correct. Can be Vec0.
        if self.t.ndim in (0, 1):
            t = expand_batch_dims(self.t, ndim=self.q.ndim - self.t.ndim)
            object.__setattr__(self, "t", t)

    # ==========================================================================
    # Array properties

    @property
    def _shape_tuple(self) -> tuple[tuple[int, ...], ComponentShapeTuple]:
        """Batch, component shape."""
        qbatch, qshape = vector_batched_shape(self.q)
        pbatch, pshape = vector_batched_shape(self.p)
        tbatch, _ = batched_shape(self.t, expect_ndim=0)
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch, tbatch)
        return batch_shape, ComponentShapeTuple(q=qshape, p=pshape, t=1)

    # ==========================================================================
    # Convenience methods

    def wt(self, *, units: Any) -> BatchVec7:
        """Phase-space position as an Array[float, (*batch, 1+Q+P)].

        This is the full phase-space position, including the time.

        Parameters
        ----------
        units : `galax.units.UnitSystem`, optional keyword-only
            The unit system If ``None``, use the current unit system.

        Returns
        -------
        wt : Array[float, (*batch, 1+Q+P)]
            The full phase-space position, including time.

        Examples
        --------
        Assuming the following imports:

        >>> from unxt import Quantity
        >>> from galax.coordinates import PhaseSpacePosition

        We can create a phase-space position and convert it to a 6-vector:

        >>> psp = PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                              p=Quantity([4, 5, 6], "km/s"),
        ...                              t=Quantity(7.0, "Myr"))
        >>> psp.wt(units="galactic")
         Array([7.00000000e+00, 1.00000000e+00, 2.00000000e+00, 3.00000000e+00,
                4.09084866e-03, 5.11356083e-03, 6.13627299e-03], dtype=float64)
        """
        usys = unitsystem(units)
        batch, comps = self._shape_tuple
        cart = self.represent_as(Cartesian3DVector)
        q = xp.broadcast_to(convert(cart.q, Quantity), (*batch, comps.q))
        p = xp.broadcast_to(convert(cart.p, Quantity), (*batch, comps.p))
        t = xp.broadcast_to(self.t.decompose(usys).value[..., None], (*batch, comps.t))
        return xp.concat((t, q.decompose(usys).value, p.decompose(usys).value), axis=-1)

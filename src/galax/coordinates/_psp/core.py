"""galax: Galactic Dynamics in Jax."""

__all__ = ["PhaseSpacePosition", "CompositePhaseSpacePosition"]

from collections.abc import Iterable
from typing import Any, NamedTuple, final

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import Array, Int, PyTree, Shaped
from typing_extensions import override

import coordinax as cx
import quaxed.array_api as xp
import quaxed.numpy as jnp
from unxt import Quantity

import galax.typing as gt
from .base import ComponentShapeTuple as BaseComponentShapeTuple
from .base_composite import AbstractCompositePhaseSpacePosition
from .base_psp import AbstractPhaseSpacePosition
from .utils import _p_converter, _q_converter
from galax.utils._misc import zeroth
from galax.utils._shape import batched_shape, expand_batch_dims, vector_batched_shape


class ComponentShapeTuple(NamedTuple):
    """Component shape of the phase-space position."""

    q: int
    """Shape of the position."""

    p: int
    """Shape of the momentum."""

    t: int | None
    """Shape of the time."""

    @classmethod
    def from_basecomponentshapetuple(
        cls, obj: BaseComponentShapeTuple, /
    ) -> "ComponentShapeTuple":
        """Create from a base component shape tuple."""
        return cls(q=obj.q, p=obj.p, t=obj.t)


def _converter_t(x: Any) -> gt.BatchableFloatQScalar | gt.FloatQScalar | None:
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

    >>> psp2 = PhaseSpacePosition(q=Quantity([1, 2, 3], "m"),
    ...                           p=Quantity([4, 5, 6], "m/s"), t=t)
    >>> psp2 == psp
    Array(True, dtype=bool)

    """

    q: cx.AbstractPosition3D = eqx.field(converter=_q_converter)
    """Positions, e.g CartesianPosition3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    p: cx.AbstractVelocity3D = eqx.field(converter=_p_converter)
    r"""Conjugate momenta, e.g. CartesianVelocity3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    t: gt.BatchableFloatQScalar | gt.FloatQScalar | None = eqx.field(
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


##############################################################################


def _concat(values: Iterable[PyTree], time_sorter: Int[Array, "..."]) -> PyTree:
    return jtu.tree_map(
        lambda *xs: xp.concat(tuple(jnp.atleast_1d(x) for x in xs), axis=-1)[
            ..., time_sorter
        ],
        *values,
    )


@final
class CompositePhaseSpacePosition(AbstractCompositePhaseSpacePosition):
    r"""Composite Phase-Space Position with time.

    The phase-space position is a point in the 7-dimensional phase space
    :math:`\mathbb{R}^7` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}`, the time :math:`t`, and the conjugate momentum
    :math:`\boldsymbol{p}`.

    This class has the same constructor semantics as `dict`.

    Parameters
    ----------
    psps: dict | tuple, optional positional-only
        initialize from a (key, value) mapping or tuple.
    **kwargs : AbstractPhaseSpacePosition
        The name=value pairs of the phase-space positions.

    Notes
    -----
    - `q`, `p`, and `t` are a concatenation of all the constituent phase-space
      positions, sorted by `t`.
    - The batch shape of `q`, `p`, and `t` are broadcast together.

    Examples
    --------
    We assume the following imports:

    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    We can create a phase-space position. Here we will use the convenience
    constructors for Cartesian positions and velocities. To see the full
    constructor, see :class:`~galax.coordinates.PhaseSpacePosition`.

    >>> psp1 = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "m"),
    ...                              p=Quantity([4, 5, 6], "m/s"),
    ...                              t=Quantity(7.0, "s"))
    >>> psp2 = gc.PhaseSpacePosition(q=Quantity([1.5, 2.5, 3.5], "m"),
    ...                              p=Quantity([4.5, 5.5, 6.5], "m/s"),
    ...                              t=Quantity(6.0, "s"))

    We can create a composite phase-space position from these two phase-space
    positions:

    >>> cpsp = gc.CompositePhaseSpacePosition(psp1=psp1, psp2=psp2)
    >>> cpsp
    CompositePhaseSpacePosition({'psp1': PhaseSpacePosition(
        q=CartesianPosition3D( ... ),
        p=CartesianVelocity3D( ... ),
        t=Quantity...
      ),
      'psp2': PhaseSpacePosition(
        q=CartesianPosition3D( ... ),
        p=CartesianVelocity3D( ... ),
        t=Quantity...
    )})

    The individual phase-space positions can be accessed via the keys:

    >>> cpsp["psp1"]
    PhaseSpacePosition(
      q=CartesianPosition3D( ... ),
      p=CartesianVelocity3D( ... ),
      t=Quantity...
    )

    The ``q``, ``p``, and ``t`` attributes are the concatenation of the
    constituent phase-space positions, sorted by ``t``. Note that in this
    example, the time of ``psp2`` is earlier than ``psp1``.

    >>> cpsp.t
    Quantity['time'](Array([6., 7.], dtype=float64), unit='s')

    >>> cpsp.q.x
    Quantity['length'](Array([1.5, 1. ], dtype=float64), unit='m')

    >>> cpsp.p.d_x
    Quantity['speed'](Array([4.5, 4. ], dtype=float64), unit='m / s')

    We can transform the composite phase-space position to a new position class.

    >>> cx.represent_as(cpsp, cx.CylindricalPosition)
    CompositePhaseSpacePosition({'psp1': PhaseSpacePosition(
        q=CylindricalPosition( ... ),
        p=CylindricalVelocity( ... ),
        t=Quantity...
      ),
      'psp2': PhaseSpacePosition(
        q=CylindricalPosition( ... ),
        p=CylindricalVelocity( ... ),
        t=...
    )})
    """

    _time_sorter: Shaped[Array, "alltimes"]
    _time_are_none: bool

    def __init__(
        self,
        psps: dict[str, AbstractPhaseSpacePosition]
        | tuple[tuple[str, AbstractPhaseSpacePosition], ...] = (),
        /,
        **kwargs: AbstractPhaseSpacePosition,
    ) -> None:
        super().__init__(psps, **kwargs)

        # TODO: check up on the shapes

        # Construct time sorter
        # Either all the times are `None` or real times
        tisnone = [psp.t is None for psp in self.values()]
        if not any(tisnone):
            ts = xp.concat([jnp.atleast_1d(w.t) for w in self.values()], axis=0)
            self._time_are_none = False
        elif all(tisnone):
            # Makes a `arange` counting up the length of each psp. For sorting,
            # 0-length psps become length 1.
            ts = jnp.cumsum(jnp.concat([jnp.ones(len(w) or 1) for w in self.values()]))
            self._time_are_none = True
        else:
            msg = "All times must be None or real times."
            raise ValueError(msg)

        self._time_sorter = xp.argsort(ts)

    @property
    def q(self) -> cx.AbstractPosition3D:
        """Positions."""
        # TODO: get AbstractPosition to work with `stack` directly
        return _concat((x.q for x in self.values()), self._time_sorter)

    @property
    def p(self) -> cx.AbstractVelocity3D:
        """Conjugate momenta."""
        # TODO: get AbstractPosition to work with `stack` directly
        return _concat((x.p for x in self.values()), self._time_sorter)

    @property
    def t(self) -> Shaped[Quantity["time"], "..."] | list[None]:
        """Times."""
        if self._time_are_none:
            return [None] * len(self._time_sorter)

        return xp.concat([jnp.atleast_1d(psp.t) for psp in self.values()], axis=0)[
            self._time_sorter
        ]

    # ==========================================================================
    # Array properties

    @override
    @property
    def _shape_tuple(self) -> tuple[gt.Shape, ComponentShapeTuple]:  # type: ignore[override]
        """Batch and component shapes."""
        batch_shape = jnp.broadcast_shapes(*[psp.shape for psp in self.values()])
        if not batch_shape:
            batch_shape = (len(self),)
        else:
            batch_shape = (*batch_shape[:-1], len(self) * batch_shape[-1])
        shape = zeroth(self.values())._shape_tuple[1]  # noqa: SLF001
        return batch_shape, ComponentShapeTuple.from_basecomponentshapetuple(shape)

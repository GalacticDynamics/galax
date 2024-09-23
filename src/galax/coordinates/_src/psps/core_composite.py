"""galax: Galactic Dynamics in Jax."""

__all__ = ["CompositePhaseSpacePosition"]

from collections.abc import Iterable
from typing import final

import jax.tree as jtu
from jaxtyping import Array, Int, PyTree, Shaped

import coordinax as cx
import quaxed.numpy as jnp
from unxt import Quantity

from .base_composite import AbstractCompositePhaseSpacePosition
from .base_psp import AbstractPhaseSpacePosition


def _concat(values: Iterable[PyTree], time_sorter: Int[Array, "..."]) -> PyTree:
    return jtu.map(
        lambda *xs: jnp.concat(tuple(jnp.atleast_1d(x) for x in xs), axis=-1)[
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

    >>> w1 = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "m"),
    ...                            p=Quantity([4, 5, 6], "m/s"),
    ...                            t=Quantity(7.0, "s"))
    >>> w2 = gc.PhaseSpacePosition(q=Quantity([1.5, 2.5, 3.5], "m"),
    ...                            p=Quantity([4.5, 5.5, 6.5], "m/s"),
    ...                            t=Quantity(6.0, "s"))

    We can create a composite phase-space position from these two phase-space
    positions:

    >>> cw = gc.CompositePhaseSpacePosition(w1=w1, w2=w2)
    >>> cw
    CompositePhaseSpacePosition({'w1': PhaseSpacePosition(
        q=CartesianPosition3D( ... ),
        p=CartesianVelocity3D( ... ),
        t=Quantity...
      ),
      'w2': PhaseSpacePosition(
        q=CartesianPosition3D( ... ),
        p=CartesianVelocity3D( ... ),
        t=Quantity...
    )})

    The individual phase-space positions can be accessed via the keys:

    >>> cw["w1"]
    PhaseSpacePosition(
      q=CartesianPosition3D( ... ),
      p=CartesianVelocity3D( ... ),
      t=Quantity...
    )

    The ``q``, ``p``, and ``t`` attributes are the concatenation of the
    constituent phase-space positions, sorted by ``t``. Note that in this
    example, the time of ``w2`` is earlier than ``w1``.

    >>> cw.t
    Quantity['time'](Array([6., 7.], dtype=float64), unit='s')

    >>> cw.q.x
    Quantity['length'](Array([1.5, 1. ], dtype=float64), unit='m')

    >>> cw.p.d_x
    Quantity['speed'](Array([4.5, 4. ], dtype=float64), unit='m / s')

    We can transform the composite phase-space position to a new position class.

    >>> cx.represent_as(cw, cx.CylindricalPosition)
    CompositePhaseSpacePosition({'w1': PhaseSpacePosition(
        q=CylindricalPosition( ... ),
        p=CylindricalVelocity( ... ),
        t=Quantity...
      ),
      'w2': PhaseSpacePosition(
        q=CylindricalPosition( ... ),
        p=CylindricalVelocity( ... ),
        t=...
    )})

    The shape of the composite phase-space position is the broadcast shape of
    the individual phase-space positions (with a minimum of 1, even when
    component has shape 0).

    >>> w1.shape, w2.shape
    ((), ())
    >>> cw.shape
    (2,)
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
            ts = jnp.concat([jnp.atleast_1d(w.t) for w in self.values()], axis=0)
            self._time_are_none = False
        elif all(tisnone):
            # Makes a `arange` counting up the length of each psp. For sorting,
            # 0-length psps become length 1.
            ts = jnp.cumsum(jnp.concat([jnp.ones(len(w) or 1) for w in self.values()]))
            self._time_are_none = True
        else:
            msg = "All times must be None or real times."
            raise ValueError(msg)

        self._time_sorter = jnp.argsort(ts)

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

        return jnp.concat([jnp.atleast_1d(psp.t) for psp in self.values()], axis=0)[
            self._time_sorter
        ]

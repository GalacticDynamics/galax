"""galax: Galactic Dynamics in Jax."""

__all__ = ["CompositePhaseSpacePosition"]

from collections.abc import Iterable, Mapping
from typing import Any, final

import jax.tree as jtu
from jaxtyping import Array, Int, PyTree, Shaped

import coordinax as cx
import coordinax.frames as cxf
import quaxed.numpy as jnp
import unxt as u
from zeroth import zeroth

from .base_composite import AbstractCompositePhaseSpacePosition
from .base_psp import AbstractPhaseSpacePosition


def _concat(values: Iterable[PyTree], time_sorter: Int[Array, "..."]) -> PyTree:
    return jtu.map(
        lambda *xs: jnp.concat(tuple(jnp.atleast_1d(x) for x in xs), axis=-1)[
            ..., time_sorter
        ],
        *values,
    )

def _to_frame_if_not_noframe(
    psp: AbstractPhaseSpacePosition, frame: cxf.AbstractReferenceFrame
) -> AbstractPhaseSpacePosition:
    return (
        psp
        if isinstance(frame, cxf.NoFrame) and isinstance(psp.frame, cxf.NoFrame)
        else psp.to_frame(frame)
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

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    We can create a phase-space position. Here we will use the convenience
    constructors for Cartesian positions and velocities. To see the full
    constructor, see :class:`~galax.coordinates.PhaseSpacePosition`.

    >>> w1 = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "m"),
    ...                            p=u.Quantity([4, 5, 6], "m/s"),
    ...                            t=u.Quantity(7, "s"))
    >>> w2 = gc.PhaseSpacePosition(q=u.Quantity([1.5, 2.5, 3.5], "m"),
    ...                            p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
    ...                            t=u.Quantity(6, "s"))

    We can create a composite phase-space position from these two phase-space
    positions:

    >>> cw = gc.CompositePhaseSpacePosition(w1=w1, w2=w2)
    >>> cw
    CompositePhaseSpacePosition({'w1': PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity...
      ),
      'w2': PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity...
    )})

    The individual phase-space positions can be accessed via the keys:

    >>> cw["w1"]
    PhaseSpacePosition(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity...
    )

    The ``q``, ``p``, and ``t`` attributes are the concatenation of the
    constituent phase-space positions, sorted by ``t``. Note that in this
    example, the time of ``w2`` is earlier than ``w1``.

    >>> cw.t
    Quantity['time'](Array([6, 7], dtype=int64, ...), unit='s')

    >>> cw.q.x
    Quantity['length'](Array([1.5, 1. ], dtype=float64), unit='m')

    >>> cw.p.d_x
    Quantity['speed'](Array([4.5, 4. ], dtype=float64), unit='m / s')

    We can transform the composite phase-space position to a new position class.

    >>> cw.vconvert(cx.vecs.CylindricalPos)
    CompositePhaseSpacePosition({'w1': PhaseSpacePosition(
        q=CylindricalPos( ... ),
        p=CylindricalVel( ... ),
        t=Quantity...
      ),
      'w2': PhaseSpacePosition(
        q=CylindricalPos( ... ),
        p=CylindricalVel( ... ),
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
    _frame: cxf.NoFrame  # TODO: support frames

    def __init__(
        self,
        psps: Mapping[str, AbstractPhaseSpacePosition]
        | tuple[tuple[str, AbstractPhaseSpacePosition], ...] = (),
        /,
        frame: Any = None,
        **kwargs: AbstractPhaseSpacePosition,
    ) -> None:
        # Aggregate all the PhaseSpacePositions
        allpsps = dict(psps, **kwargs)

        # Everything must be transformed to be in the same frame.
        # Compute and store the frame
        maybeframe = frame if frame is not None else zeroth(allpsps.values()).frame
        frame = (
            maybeframe
            if isinstance(maybeframe, cxf.AbstractReferenceFrame)
            else cxf.TransformedReferenceFrame.from_(maybeframe)
        )
        self._frame = frame
        # Transform all the PhaseSpacePositions to that frame. If the frames are
        # already `NoFrame`, we can skip this step, since no transformation is
        # possible in `NoFrame`.
        allpsps = {k: _to_frame_if_not_noframe(psp) for k, psp in allpsps.items()}

        # Now we can set all the PhaseSpacePositions
        super().__init__(allpsps)

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
    def q(self) -> cx.vecs.AbstractPos3D:
        """Positions."""
        # TODO: get AbstractPos to work with `stack` directly
        return _concat((x.q for x in self.values()), self._time_sorter)

    @property
    def p(self) -> cx.vecs.AbstractVel3D:
        """Conjugate momenta."""
        # TODO: get AbstractPos to work with `stack` directly
        return _concat((x.p for x in self.values()), self._time_sorter)

    @property
    def t(self) -> Shaped[u.Quantity["time"], "..."] | list[None]:
        """Times."""
        if self._time_are_none:
            return [None] * len(self._time_sorter)

        return jnp.concat([jnp.atleast_1d(psp.t) for psp in self.values()], axis=0)[
            self._time_sorter
        ]

    @property
    def frame(self) -> cxf.AbstractReferenceFrame:
        """The reference frame."""
        return self._frame

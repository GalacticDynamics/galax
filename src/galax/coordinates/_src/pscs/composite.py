"""galax: Galactic Dynamics in Jax."""

__all__ = ["CompositePhaseSpaceCoordinate"]

from collections.abc import Iterable, Mapping
from typing import Any, final

import jax.tree as jtu
from jaxtyping import Array, Int, PyTree, Shaped

import coordinax as cx
import coordinax.frames as cxf
import quaxed.numpy as jnp
import unxt as u
from zeroth import zeroth

from .base_composite import AbstractCompositePhaseSpaceCoordinate
from .base_single import AbstractBasicPhaseSpaceCoordinate
from galax.coordinates._src.frames import SimulationFrame


def _concat(values: Iterable[PyTree], time_sorter: Int[Array, "..."]) -> PyTree:
    return jtu.map(
        lambda *xs: jnp.concat(tuple(jnp.atleast_1d(x) for x in xs), axis=-1)[
            ..., time_sorter
        ],
        *values,
    )


@final
class CompositePhaseSpaceCoordinate(AbstractCompositePhaseSpaceCoordinate):
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
    **kwargs : AbstractBasicPhaseSpaceCoordinate
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
    constructor, see :class:`~galax.coordinates.PhaseSpaceCoordinate`.

    >>> w1 = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "m"),
    ...                            p=u.Quantity([4, 5, 6], "m/s"),
    ...                            t=u.Quantity(7, "s"))
    >>> w2 = gc.PhaseSpaceCoordinate(q=u.Quantity([1.5, 2.5, 3.5], "m"),
    ...                            p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
    ...                            t=u.Quantity(6, "s"))

    We can create a composite phase-space position from these two phase-space
    positions:

    >>> cw = gc.CompositePhaseSpaceCoordinate(w1=w1, w2=w2)
    >>> cw
    CompositePhaseSpaceCoordinate({'w1': PhaseSpaceCoordinate(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity...
      ),
      'w2': PhaseSpaceCoordinate(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity...
    )})

    The individual phase-space positions can be accessed via the keys:

    >>> cw["w1"]
    PhaseSpaceCoordinate(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity...
    )

    The ``q``, ``p``, and ``t`` attributes are the concatenation of the
    constituent phase-space positions, sorted by ``t``. Note that in this
    example, the time of ``w2`` is earlier than ``w1``.

    >>> cw.t
    Quantity(Array([6, 7], dtype=int64, ...), unit='s')

    >>> cw.q.x
    Quantity(Array([1.5, 1. ], dtype=float64), unit='m')

    >>> cw.p.x
    Quantity(Array([4.5, 4. ], dtype=float64), unit='m / s')

    We can transform the composite phase-space position to a new position class.

    >>> cw.vconvert(cx.vecs.CylindricalPos)
    CompositePhaseSpaceCoordinate({'w1': PhaseSpaceCoordinate(
        q=CylindricalPos( ... ),
        p=CylindricalVel( ... ),
        t=Quantity...
      ),
      'w2': PhaseSpaceCoordinate(
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
    _frame: SimulationFrame  # TODO: support frames

    def __init__(
        self,
        psps: Mapping[str, AbstractBasicPhaseSpaceCoordinate]
        | tuple[tuple[str, AbstractBasicPhaseSpaceCoordinate], ...] = (),
        /,
        frame: Any = None,
        **kwargs: AbstractBasicPhaseSpaceCoordinate,
    ) -> None:
        # Aggregate all the PhaseSpaceCoordinates
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
        # Transform all the PhaseSpaceCoordinates to that frame. If the frames are
        # already `NoFrame`, we can skip this step, since no transformation is
        # possible in `NoFrame`.
        allpsps = {k: psp.to_frame(frame) for k, psp in allpsps.items()}

        # Now we can set all the PhaseSpaceCoordinates
        super().__init__(allpsps)

        # TODO: check up on the shapes

        # Construct time sorter TODO: fix unxt primitive for jnp.concat to do
        # type coercion after ustrip, which can change types.
        ts = jnp.concat(
            [jnp.atleast_1d(w.t).astype(float) for w in self.values()], axis=0
        )
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
    def t(self) -> Shaped[u.Quantity["time"], "..."]:
        """Times."""
        # TODO: sort during, not after
        return jnp.concat([jnp.atleast_1d(wt.t) for wt in self.values()], axis=0)[
            self._time_sorter
        ]

    @property
    def frame(self) -> cxf.AbstractReferenceFrame:
        """The reference frame."""
        return self._frame

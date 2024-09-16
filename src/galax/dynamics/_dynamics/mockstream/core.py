"""Mock stellar streams."""

__all__ = ["MockStreamArm", "MockStream"]

from dataclasses import replace
from typing import TYPE_CHECKING, Any, final

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jtu
from jaxtyping import Array, Shaped

import coordinax as cx
import quaxed.array_api as xp
from unxt import Quantity

import galax.typing as gt
from galax.coordinates import (
    AbstractCompositePhaseSpacePosition,
    AbstractPhaseSpacePosition,
    ComponentShapeTuple,
)
from galax.coordinates._psp.utils import getitem_vec1time_index
from galax.utils._shape import batched_shape, vector_batched_shape

if TYPE_CHECKING:
    from typing import Self


@final
class MockStreamArm(AbstractPhaseSpacePosition):
    """Component of a mock stream object.

    Parameters
    ----------
    q : Array[float, (*batch, 3)]
        Positions (x, y, z).
    p : Array[float, (*batch, 3)]
        Conjugate momenta (v_x, v_y, v_z).
    t : Array[float, (*batch,)]
        Array of times corresponding to the positions.
    release_time : Array[float, (*batch,)]
        Release time of the stream particles [Myr].
    """

    q: cx.AbstractPosition3D = eqx.field(converter=cx.AbstractPosition3D.constructor)
    """Positions (x, y, z)."""

    p: cx.AbstractVelocity3D = eqx.field(converter=cx.AbstractVelocity3D.constructor)
    r"""Conjugate momenta (v_x, v_y, v_z)."""

    t: gt.QVecTime = eqx.field(converter=Quantity["time"].constructor)
    """Array of times corresponding to the positions."""

    release_time: gt.QVecTime = eqx.field(converter=Quantity["time"].constructor)
    """Release time of the stream particles [Myr]."""

    # ==========================================================================
    # Array properties

    @property
    def _shape_tuple(self) -> tuple[gt.Shape, ComponentShapeTuple]:
        """Batch ."""
        qbatch, qshape = vector_batched_shape(self.q)
        pbatch, pshape = vector_batched_shape(self.p)
        tbatch, _ = batched_shape(self.t, expect_ndim=0)
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch, tbatch)
        return batch_shape, ComponentShapeTuple(q=qshape, p=pshape, t=1)

    def __getitem__(self, index: Any) -> "Self":
        """Return a new object with the given slice applied."""
        # Compute subindex
        subindex = getitem_vec1time_index(index, self.t)
        # Apply slice
        return replace(
            self,
            q=self.q[index],
            p=self.p[index],
            t=self.t[subindex],
            release_time=self.release_time[subindex],
        )


##############################################################################


@final
class MockStream(AbstractCompositePhaseSpacePosition):
    _time_sorter: Shaped[Array, "alltimes"]

    def __init__(
        self,
        psps: dict[str, MockStreamArm] | tuple[tuple[str, MockStreamArm], ...] = (),
        /,
        **kwargs: MockStreamArm,
    ) -> None:
        super().__init__(psps, **kwargs)

        # TODO: check up on the shapes

        # Construct time sorter
        ts = xp.concat([psp.release_time for psp in self.values()], axis=0)
        self._time_sorter = xp.argsort(ts)

    @property
    def q(self) -> cx.AbstractPosition3D:
        """Positions."""
        # TODO: get AbstractPosition to work with `stack` directly
        return jtu.map(
            lambda *x: xp.concat(x, axis=-1)[..., self._time_sorter],
            *(x.q for x in self.values()),
        )

    @property
    def p(self) -> cx.AbstractVelocity3D:
        """Conjugate momenta."""
        # TODO: get AbstractVelocity to work with `stack` directly
        return jtu.map(
            lambda *x: xp.concat(x, axis=-1)[..., self._time_sorter],
            *(x.p for x in self.values()),
        )

    @property
    def t(self) -> Shaped[Quantity["time"], "..."]:
        """Times."""
        return xp.concat([psp.t for psp in self.values()], axis=0)[self._time_sorter]

    @property
    def release_time(self) -> Shaped[Quantity["time"], "..."]:
        """Release times."""
        return xp.concat([psp.release_time for psp in self.values()], axis=0)[
            self._time_sorter
        ]

"""Mock stellar streams."""

__all__ = ["MockStream"]

from typing import final

import jax.tree as jtu
from jaxtyping import Array, Shaped

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from zeroth import zeroth

import galax.coordinates as gc
from .arm import MockStreamArm


@final
class MockStream(gc.AbstractCompositePhaseSpaceCoordinate):
    """Mock Stellar Stream."""

    _time_sorter: Shaped[Array, "alltimes"]
    _frame: gc.frames.SimulationFrame  # TODO: support frames

    def __init__(
        self,
        psps: dict[str, MockStreamArm] | tuple[tuple[str, MockStreamArm], ...] = (),
        /,
        **kwargs: MockStreamArm,
    ) -> None:
        # Aggregate all the MockStreamArm
        allpsps = dict(psps, **kwargs)

        # Everything must be transformed to be in the same frame.
        # Compute and store the frame
        self._frame = theframe = zeroth(allpsps.values()).frame
        # Transform all the PhaseSpaceCoordinates to that frame. If the frames
        # are already `NoFrame`, we can skip this step, since no transformation
        # is possible in `NoFrame`.
        allpsps = {k: psp.to_frame(theframe) for k, psp in allpsps.items()}

        super().__init__(psps, **kwargs)

        # TODO: check up on the shapes

        # Construct time sorter
        ts = jnp.concat([psp.release_time for psp in self.values()], axis=0)
        self._time_sorter = jnp.argsort(ts)

    @property
    def q(self) -> cx.vecs.AbstractPos3D:
        """Positions."""
        # TODO: get AbstractPos to work with `stack` directly
        return jtu.map(
            lambda *x: jnp.concat(x, axis=-1)[..., self._time_sorter],
            *(x.q for x in self.values()),
        )

    @property
    def p(self) -> cx.vecs.AbstractVel3D:
        """Conjugate momenta."""
        # TODO: get AbstractVel to work with `stack` directly
        return jtu.map(
            lambda *x: jnp.concat(x, axis=-1)[..., self._time_sorter],
            *(x.p for x in self.values()),
        )

    @property
    def t(self) -> Shaped[u.Quantity["time"], "..."]:
        """Times."""
        return jnp.concat([psp.t for psp in self.values()], axis=0)[self._time_sorter]

    @property
    def release_time(self) -> Shaped[u.Quantity["time"], "..."]:
        """Release times."""
        return jnp.concat([psp.release_time for psp in self.values()], axis=0)[
            self._time_sorter
        ]

    @property
    def frame(self) -> cx.frames.AbstractReferenceFrame:
        """The reference frame of the phase-space position."""
        return self._frame

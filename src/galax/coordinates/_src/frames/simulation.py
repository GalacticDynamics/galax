"""Simulation Reference Frames.

Building off of `coordinax.frames`.

"""

__all__ = ["SimulationFrame", "simulation_frame"]

from typing import final

from plum import dispatch

import coordinax as cx
from oncequinox import SingletonModuleMeta


@final
class SimulationFrame(cx.frames.AbstractReferenceFrame, metaclass=SingletonModuleMeta):  # type: ignore[misc]
    """The simulation reference frame.

    This is a reference frame that cannot be transformed to or from.

    Examples
    --------
    >>> import coordinax.frames as cxf
    >>> import galax.coordinates as gc

    >>> sim = gc.frames.SimulationFrame()

    >>> cxf.frame_transform_op(sim, sim)
    Identity()

    >>> icrs = cxf.ICRS()
    >>> try:
    ...     cxf.frame_transform_op(sim, icrs)
    ... except Exception as e:
    ...     print(e)
    `frame_transform_op(SimulationFrame(), ICRS())` could not be resolved...

    """


simulation_frame = SimulationFrame()


@dispatch
def frame_transform_op(
    from_frame: SimulationFrame,  # noqa: ARG001
    to_frame: SimulationFrame,  # noqa: ARG001
    /,
) -> cx.ops.Identity:
    """Return an identity operator for the Simulation->Simulation transformation.

    Examples
    --------
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> sim_frame = gc.frames.SimulationFrame()
    >>> frame_op = cx.frames.frame_transform_op(sim_frame, sim_frame)
    >>> frame_op
    Identity()

    """
    return cx.ops.Identity()

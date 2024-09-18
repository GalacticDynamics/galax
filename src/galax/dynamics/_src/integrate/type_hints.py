__all__ = ["VectorField"]

from typing import Any, Protocol, runtime_checkable

import galax.typing as gt


@runtime_checkable
class VectorField(Protocol):
    """Protocol for the integration callable."""

    def __call__(self, t: gt.FloatScalar, w: gt.Vec6, args: tuple[Any, ...]) -> gt.Vec6:
        """Integration function.

        Parameters
        ----------
        t : float
            The time. This is the integration variable.
        w : Array[float, (6,)]
            The position and velocity.
        args : tuple[Any, ...]
            Additional arguments.

        Returns
        -------
        Array[float, (6,)]
            Velocity and acceleration [v (3,), a (3,)].
        """
        ...

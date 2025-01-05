__all__ = ["VectorField"]

from typing import Any, Protocol, runtime_checkable

import galax.typing as gt


@runtime_checkable
class VectorField(Protocol):
    """Protocol for the integration callable."""

    def __call__(
        self, t: gt.FloatScalar, w: gt.QParr, args: tuple[Any, ...]
    ) -> gt.PAarr:
        """Integration function.

        Parameters
        ----------
        t : float
            The time. This is the integration variable.
        qp : Array[number, (3,)], Array[number, (3,)]
            The position and velocity.
        args : tuple[Any, ...]
            Additional arguments.

        Returns
        -------
        Array[number, (3,)], Array[number, (3,)]
            Velocity and acceleration p, a.

        """
        ...

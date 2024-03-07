"""Moving object. TEMPORARY CLASS.

This class adds time-dependent translations to a potential.
It is NOT careful about the implied changes to velocity, etc.
"""

__all__: list[str] = []


from collections.abc import Callable
from typing import Literal, final

import equinox as eqx

from coordinax.operators import AbstractOperator

from galax.typing import FloatScalar, RealScalar, Vec3


@final
class TimeDependentSpatialTranslationOperator(AbstractOperator):  # type: ignore[misc]
    r"""Operator for time-dependent translation."""

    translation: Callable[[FloatScalar], Vec3] = eqx.field()
    """The spatial translation."""

    def __call__(self, q: Vec3, t: RealScalar) -> tuple[Vec3, RealScalar]:
        """Do."""
        return (q + self.translation(t), t)

    @property
    def is_inertial(self) -> Literal[False]:
        """Galilean translation is an inertial frame-preserving transformation."""
        return False

    @property
    def inverse(self) -> "TimeDependentSpatialTranslationOperator":
        """The inverse of the operator."""
        return TimeDependentSpatialTranslationOperator(
            translation=lambda t: -self.translation(t)
        )

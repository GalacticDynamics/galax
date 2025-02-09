"""Phase-space positions.

This is private API.

"""

__all__ = [
    # Base
    "AbstractPhaseSpaceCoordinate",
    "ComponentShapeTuple",
    # Single
    "AbstractBasicPhaseSpaceCoordinate",
    "PhaseSpaceCoordinate",
    # Composite
    "AbstractCompositePhaseSpaceCoordinate",
    "CompositePhaseSpaceCoordinate",
]

from .base import AbstractPhaseSpaceCoordinate, ComponentShapeTuple
from .base_composite import AbstractCompositePhaseSpaceCoordinate
from .base_single import AbstractBasicPhaseSpaceCoordinate
from .composite import CompositePhaseSpaceCoordinate
from .single import PhaseSpaceCoordinate

# Register by import
# isort: split
from . import (
    register_primitives,  # noqa: F401
    register_vectorapi,  # noqa: F401
)

"""Phase-space positions.

This is private API.

"""

__all__ = ["PhaseSpacePosition", "ComponentShapeTuple"]

from .core import ComponentShapeTuple, PhaseSpacePosition

# Register by import
# isort: split
from . import (
    register_primitives,  # noqa: F401
    register_vectorapi,  # noqa: F401
)

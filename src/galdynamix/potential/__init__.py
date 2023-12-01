"""galdynamix: Galactic Dynamix in Jax."""

from __future__ import annotations

from . import _potential
from ._potential import *

__all__: list[str] = []
__all__ += _potential.__all__

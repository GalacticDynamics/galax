"""``galax`` Potentials."""
# ruff:noqa: F401

from . import builtin, nfw
from .builtin import *
from .nfw import *

__all__: list[str] = []
__all__ += builtin.__all__
__all__ += nfw.__all__

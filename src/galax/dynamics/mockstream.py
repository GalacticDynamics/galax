""":mod:`galax.dynamics.mockstream`."""

from ._src import mockstream
from ._src.mockstream import *  # noqa: F403

__all__: list[str] = []
__all__ += mockstream.__all__

# Cleanup
del mockstream

"""Plotting Utils.

TODO: move this to a separate package.

"""

__all__ = [
    "AbstractPlottingBackend",
    "MatplotlibBackend",
]

from typing import final


class AbstractPlottingBackend:
    """Abstract base class for plotting backends."""


@final
class MatplotlibBackend(AbstractPlottingBackend):
    """Matplotlib plotting backend."""

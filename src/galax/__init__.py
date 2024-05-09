"""Copyright (c) 2023 galax maintainers. All rights reserved."""

__all__ = [
    "__version__",
    "__version_tuple__",
    # Modules
    "coordinates",
    "potential",
    "dynamics",
    "utils",
    "typing",
]

from . import (
    coordinates,
    dynamics,
    potential,
    setup_package as setup_package,
    typing,
    utils,
)
from ._version import __version__, __version_tuple__

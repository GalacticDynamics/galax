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

from . import setup_package as setup_package

# isort: split
from . import (
    coordinates,
    dynamics,
    potential,
    typing,
    utils,
)
from ._version import version as __version__, version_tuple as __version_tuple__

# Optional dependencies
# The act of importing registers in the functionality
# isort: split
from ._interop import *

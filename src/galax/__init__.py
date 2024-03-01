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

from jax import config

from . import coordinates, dynamics, potential, typing, utils
from ._version import __version__, __version_tuple__

config.update("jax_enable_x64", True)  # noqa: FBT003

del config

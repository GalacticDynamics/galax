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
from ._version import version as __version__, version_tuple as __version_tuple__

if utils._optional_deps.HAS_ASTROPY:  # noqa: SLF001
    from ._interop import galax_interop_astropy  # noqa: F401

if utils._optional_deps.HAS_GALA:  # noqa: SLF001
    from ._interop import galax_interop_gala  # noqa: F401

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

# ---------------------------
# Optional dependencies
# The act of importing registers in the functionality

if utils._optional_deps.HAS_ASTROPY:  # noqa: SLF001
    from ._interop import galax_interop_astropy

    del galax_interop_astropy

if utils._optional_deps.HAS_GALA:  # noqa: SLF001
    from ._interop import galax_interop_gala

    del galax_interop_gala

if utils._optional_deps.HAS_GALPY:  # noqa: SLF001
    from ._interop import galax_interop_galpy

    del galax_interop_galpy

if utils._optional_deps.HAS_MATPLOTLIB:  # noqa: SLF001
    from ._interop import matplotlib

    del matplotlib

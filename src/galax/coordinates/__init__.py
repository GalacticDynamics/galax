""":mod:`galax.coordinates` --- Coordinate systems and transformations.

Copyright (c) 2023 galax maintainers. All rights reserved.
"""

__all__ = [
    # Modules
    "ops",
    "frames",
    # Base
    "AbstractPhaseSpaceObject",
    # Coordinates
    "AbstractBasicPhaseSpaceCoordinate",
    "AbstractPhaseSpaceCoordinate",
    "PhaseSpaceCoordinate",
    "AbstractCompositePhaseSpaceCoordinate",
    "CompositePhaseSpaceCoordinate",
    "ComponentShapeTuple",
    # PSPs
    "PhaseSpacePosition",
    "PSPComponentShapeTuple",
    # Protocols
    "PhaseSpaceObjectInterpolant",
]

from .setup_package import install_import_hook, load_interop_plugins

with install_import_hook("galax.coordinates"):
    from . import frames, ops
    from ._src.base import AbstractPhaseSpaceObject
    from ._src.interp import PhaseSpaceObjectInterpolant
    from ._src.pscs import (
        AbstractBasicPhaseSpaceCoordinate,
        AbstractCompositePhaseSpaceCoordinate,
        AbstractPhaseSpaceCoordinate,
        ComponentShapeTuple,
        CompositePhaseSpaceCoordinate,
        PhaseSpaceCoordinate,
    )
    from ._src.psps import (
        ComponentShapeTuple as PSPComponentShapeTuple,
        PhaseSpacePosition,
    )

# Clean up the namespace
del install_import_hook

# Interoperability with third-party libraries. Importing a registered module is
# what performs its `plum` dispatch registration; entry points let separately
# installed distributions extend `galax.coordinates` without it knowing they exist.
load_interop_plugins("galax.coordinates.interop")

del load_interop_plugins

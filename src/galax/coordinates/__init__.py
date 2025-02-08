""":mod:`galax.coordinates` --- Coordinate systems and transformations.

Copyright (c) 2023 galax maintainers. All rights reserved.
"""

__all__ = [
    # Modules
    "ops",
    "frames",
    # Contents
    "AbstractPhaseSpaceObject",
    # PSPs
    "AbstractPhaseSpacePosition",
    "AbstractOnePhaseSpacePosition",
    "PhaseSpacePosition",
    "AbstractCompositePhaseSpacePosition",
    "CompositePhaseSpacePosition",
    # Utils
    "ComponentShapeTuple",
    # Protocols
    "PhaseSpaceObjectInterpolant",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.coordinates", RUNTIME_TYPECHECKER):
    from . import frames, ops
    from ._src.base import AbstractPhaseSpaceObject
    from ._src.interp import PhaseSpaceObjectInterpolant
    from ._src.psps import (
        AbstractCompositePhaseSpacePosition,
        AbstractOnePhaseSpacePosition,
        AbstractPhaseSpacePosition,
        ComponentShapeTuple,
        CompositePhaseSpacePosition,
        PhaseSpacePosition,
    )

# Clean up the namespace
del install_import_hook, RUNTIME_TYPECHECKER

""":mod:`galax.coordinates` --- Coordinate systems and transformations.

Copyright (c) 2023 galax maintainers. All rights reserved.
"""

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER
from galax.utils._optional_deps import HAS_GALA

with install_import_hook("galax.coordinates", RUNTIME_TYPECHECKER):
    from . import operators
    from ._psp import *

from . import _psp  # only for __all__

__all__: list[str] = ["operators"]
__all__ += _psp.__all__

# Clean up the namespace
del install_import_hook, RUNTIME_TYPECHECKER, HAS_GALA, _psp

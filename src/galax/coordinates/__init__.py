""":mod:`galax.coordinates` --- Coordinate systems and transformations.

Copyright (c) 2023 galax maintainers. All rights reserved.
"""

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.coordinates", RUNTIME_TYPECHECKER):
    from . import ops
    from ._src import psps
    from ._src.psps import operator_compat
    from ._src.psps.base import *
    from ._src.psps.base_composite import *
    from ._src.psps.base_psp import *
    from ._src.psps.core import *
    from ._src.psps.core_composite import *
    from ._src.psps.interp import *
    from ._src.psps.utils import *

__all__: list[str] = ["ops"]
__all__ += psps.base.__all__
__all__ += psps.base_psp.__all__
__all__ += psps.base_composite.__all__
__all__ += psps.core.__all__
__all__ += psps.core_composite.__all__
__all__ += psps.interp.__all__
__all__ += psps.utils.__all__

# Clean up the namespace
del install_import_hook, RUNTIME_TYPECHECKER, psps, operator_compat

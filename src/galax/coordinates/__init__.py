""":mod:`galax.coordinates` --- Coordinate systems and transformations.

Copyright (c) 2023 galax maintainers. All rights reserved.
"""

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER
from galax.utils._optional_deps import HAS_GALA

with install_import_hook("galax.coordinates", RUNTIME_TYPECHECKER):
    from . import _psp, operators
    from ._psp.base import *
    from ._psp.base_composite import *
    from ._psp.base_psp import *
    from ._psp.core import *
    from ._psp.interp import *
    from ._psp.utils import *

__all__: list[str] = ["operators"]
__all__ += _psp.base.__all__
__all__ += _psp.base_psp.__all__
__all__ += _psp.base_composite.__all__
__all__ += _psp.core.__all__
__all__ += _psp.interp.__all__
__all__ += _psp.utils.__all__

# Clean up the namespace
del install_import_hook, RUNTIME_TYPECHECKER, HAS_GALA, _psp

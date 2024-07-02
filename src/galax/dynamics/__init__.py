""":mod:`galax.dynamics`."""

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER
from galax.utils._optional_deps import HAS_GALA

with install_import_hook("galax.dynamics", RUNTIME_TYPECHECKER):
    from . import _dynamics
    from ._dynamics import *

__all__ = _dynamics.__all__

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER, HAS_GALA, _dynamics

""":mod:`galax.dynamics`."""

from jaxtyping import install_import_hook
from lazy_loader import attach_stub

from galax.setup_package import RUNTIME_TYPECHECKER
from galax.utils._optional_deps import HAS_GALA

with install_import_hook("galax.dynamics", RUNTIME_TYPECHECKER):
    __getattr__, __dir__, __all__ = attach_stub(__name__, __file__)

if HAS_GALA:
    from . import _compat  # noqa: F401


# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER, HAS_GALA

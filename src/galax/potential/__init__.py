""":mod:`galax.potential`."""

from jaxtyping import install_import_hook
from lazy_loader import attach_stub

from galax.setup_package import RUNTIME_TYPECHECKER

__getattr__, __dir__, __all__ = attach_stub(__name__, __file__)

# install_import_hook("galax.potential", RUNTIME_TYPECHECKER)  # noqa: ERA001

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

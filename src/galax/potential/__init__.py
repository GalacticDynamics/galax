""":mod:`galax.potential`."""

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.potential", RUNTIME_TYPECHECKER):
    from ._potential import *

from . import _potential  # only for __all__

__all__: list[str] = []
__all__ += _potential.__all__

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER

"""Copyright (c) 2023 galax maintainers. All rights reserved."""
# ruff:noqa: F401

import os

from jax import config
from jaxtyping import install_import_hook
from lazy_loader import attach_stub as _attach_stub

config.update("jax_enable_x64", True)  # noqa: FBT003

_RUNTIME_TYPECHECKER = (
    "beartype.beartype"
    if (os.environ.get("GALAX_ENABLE_RUNTIME_TYPECHECKS", "1") == "1")
    else None
)

with install_import_hook("galax", _RUNTIME_TYPECHECKER):
    __getattr__, __dir__, __all__ = _attach_stub(__name__, __file__)


# Install the runtime typechecker
install_import_hook("galax", _RUNTIME_TYPECHECKER)

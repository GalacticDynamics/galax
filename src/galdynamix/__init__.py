"""Copyright (c) 2023 galdynamix maintainers. All rights reserved."""

__all__ = ["__version__"]

import os

from jax import config
from jaxtyping import install_import_hook

from ._version import version as __version__

config.update("jax_enable_x64", True)  # noqa: FBT003

if os.environ.get("GALDYNAMIX_ENABLE_RUNTIME_TYPECHECKS", "1") == "1":
    install_import_hook(["galdynamix"], "beartype.beartype")

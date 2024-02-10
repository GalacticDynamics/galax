"""Copyright (c) 2023 galax maintainers. All rights reserved."""

import os

from jax import config

config.update("jax_enable_x64", True)  # noqa: FBT003

RUNTIME_TYPECHECKER = (
    "beartype.beartype"
    if (os.environ.get("GALAX_ENABLE_RUNTIME_TYPECHECKS", "1") == "1")
    else None
)

"""Galax package setup.

Copyright (c) 2023 galax maintainers. All rights reserved.
"""

import os
from typing import Final

from jax import config

config.update("jax_enable_x64", True)  # noqa: FBT003


RUNTIME_TYPECHECKER: Final[str | None] = (
    v
    if (v := os.environ.get("GALAX_ENABLE_RUNTIME_TYPECHECKING", None)) != "None"
    else None
)
"""Runtime type checking variable "GALAX_ENABLE_RUNTIME_TYPECHECKING".

Set to "None" to disable runtime typechecking (default). Set to
"beartype.beartype" to enable runtime typechecking.

See https://docs.kidger.site/jaxtyping/api/runtime-type-checking for more
information on options.

"""

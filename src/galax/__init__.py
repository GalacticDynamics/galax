"""Copyright (c) 2023 galax maintainers. All rights reserved."""
# ruff:noqa: F401

__all__ = [
    "__version__",
    # modules
    "units",
    "potential",
    "integrator",
    "dynamics",
    "utils",
    "typing",
]

import os

from jax import config
from jaxtyping import install_import_hook

from ._version import version as __version__

config.update("jax_enable_x64", True)  # noqa: FBT003

typechecker: str | None
if os.environ.get("GALDYNAMIX_ENABLE_RUNTIME_TYPECHECKS", "1") == "1":
    typechecker = "beartype.beartype"
else:
    typechecker = None

with install_import_hook("galax", typechecker):
    from galax import dynamics, integrate, potential, typing, units, utils

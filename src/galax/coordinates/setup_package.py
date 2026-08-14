"""`galax.coordinates` package setup.

Copyright (c) 2023 galax maintainers. All rights reserved.
"""

__all__: tuple[str, ...] = ()

import contextlib
import os
from importlib.metadata import entry_points

from collections.abc import Sequence
from jaxtyping import install_import_hook as _install_import_hook
from typing import Any, Final, Literal

_RUNTIME_TYPECHECKER: str | None | Literal[False]
match os.getenv("GALAX_ENABLE_RUNTIME_TYPECHECKING", "False"):
    case "False":
        _RUNTIME_TYPECHECKER = False
    case "None":
        _RUNTIME_TYPECHECKER = None
    case str() as _name:
        _RUNTIME_TYPECHECKER = _name

RUNTIME_TYPECHECKER: Final[str | None | Literal[False]] = _RUNTIME_TYPECHECKER
"""Runtime type checking variable "GALAX_ENABLE_RUNTIME_TYPECHECKING".

Set to "False" to disable runtime typechecking (default).
Set to "None" to only enable typechecking for `@jaxtyped`-decorated functions.
Set to "beartype.beartype" to enable runtime typechecking.

See https://docs.kidger.site/jaxtyping/api/runtime-type-checking for more
information on options.

"""


def install_import_hook(
    modules: str | Sequence[str], /
) -> contextlib.AbstractContextManager[Any, None]:
    """Install the jaxtyping import hook for the given modules.

    Parameters
    ----------
    modules
        Module name or sequence of module names to install the import hook for.

    Returns
    -------
    contextlib.AbstractContextManager
        Context manager that installs the import hook on entry and removes it on exit.

    """
    return (
        _install_import_hook(modules, RUNTIME_TYPECHECKER)
        if RUNTIME_TYPECHECKER is not False
        else contextlib.nullcontext()
    )


def load_interop_plugins(group: str, /) -> None:
    """Import every module registered under an interop entry-point group.

    Interoperability distributions register their registration modules against a
    portion's group; importing such a module is what performs the `plum` dispatch
    registration. This is the only mechanism that lets a portion pick up an
    extension distribution that did not exist when the portion was written.

    A plugin whose third-party target is not installed is skipped. A
    `ModuleNotFoundError` naming a `galax` module is a real bug, not a missing
    optional dependency, so it propagates.

    Parameters
    ----------
    group
        Entry-point group to load, e.g. ``"galax.potential.interop"``.

    """
    for ep in entry_points(group=group):
        try:
            ep.load()
        except ModuleNotFoundError as e:
            if (e.name or "").split(".")[0] == "galax":
                raise

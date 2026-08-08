"""Doctest configuration."""

import importlib
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from pathlib import Path

from types import ModuleType
from typing import Any

import sybil.document
from sybil import Sybil
from sybil.parsers import myst, rest
from sybil.sybil import SybilCollection

from optional_dependencies import OptionalDependencyEnum, auto
from optional_dependencies.utils import chain_checks, get_version, is_installed

_SRC = (Path(__file__).parent / "src").resolve()
_sybil_import_path = sybil.document.import_path


def _import_path(path: Path) -> ModuleType:
    """Import a source file as its real module, for namespace packages.

    Sybil derives a module name by walking up from the file until it finds a
    directory without `__init__.py` (`sybil.python.import_path`). `src/galax` is
    a PEP 420 namespace directory, so that walk stops one level too deep and
    yields `potential._src.api` instead of `galax.potential._src.api`. Resolving
    against `src/` gives the true dotted name. Anything outside `src/` falls
    back to Sybil's own logic.

    This is separate from pytest's `--import-mode`, which Sybil does not use.
    """
    try:
        relative = Path(path).resolve().relative_to(_SRC)
    except ValueError:
        return _sybil_import_path(path)
    parts = relative.parts[:-1]
    if relative.name != "__init__.py":
        parts += (relative.stem,)
    return importlib.import_module(".".join(parts))


sybil.document.import_path = _import_path

optionflags = ELLIPSIS | NORMALIZE_WHITESPACE

parsers = [
    myst.DocTestDirectiveParser(optionflags=optionflags),
    myst.PythonCodeBlockParser(doctest_optionflags=optionflags),
    myst.SkipParser(),
]

docs = Sybil(parsers=parsers, patterns=["*.md"])
python = Sybil(  # TODO: get working with myst parsers
    parsers=[
        rest.DocTestParser(optionflags=optionflags),
        rest.PythonCodeBlockParser(),
        rest.SkipParser(),
    ],
    patterns=["*.py"],
)
rst_docs = Sybil(  # TODO: deprecate
    parsers=[
        rest.DocTestParser(optionflags=optionflags),
        rest.PythonCodeBlockParser(),
        rest.SkipParser(),
    ],
    patterns=["*.rst", "*.py"],
)

pytest_collect_file = SybilCollection((docs, python, rst_docs)).pytest()


class OptDeps(OptionalDependencyEnum):
    """Optional dependencies for ``galax``."""

    ASTROPY = auto()
    GALA = chain_checks(get_version("gala"), is_installed("gala.dynamics"))
    GALPY = auto()
    MATPLOTLIB = auto()


collect_ignore_glob = []
if not OptDeps.ASTROPY.installed:
    collect_ignore_glob.append("src/galax/interop/astropy/*")
if not OptDeps.GALA.installed:
    collect_ignore_glob.append("src/galax/interop/gala/*")
if not OptDeps.GALPY.installed:
    collect_ignore_glob.append("src/galax/interop/galpy/*")


def pytest_report_header(config: Any) -> str:  # noqa: D103, ARG001
    hdr = []

    if OptDeps.ASTROPY.installed:
        hdr.append(f"astropy: {get_version('astropy')}")
    if OptDeps.GALA.installed:
        hdr.append(f"gala: {get_version('gala')}")
    if OptDeps.GALPY.installed:
        hdr.append(f"galpy: {get_version('galpy')}")

    return "\n".join(hdr)

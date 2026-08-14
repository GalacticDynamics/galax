"""Doctest configuration."""

import importlib
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from pathlib import Path

from typing import Any

from sybil import Sybil
from sybil.document import PythonDocStringDocument
from sybil.example import Example, NotEvaluated
from sybil.parsers import myst, rest
from sybil.sybil import SybilCollection

from optional_dependencies import OptionalDependencyEnum, auto
from optional_dependencies.utils import chain_checks, get_version, is_installed

_SRC = (Path(__file__).parent / "src").resolve()


class NamespacePackageDocument(PythonDocStringDocument):
    """Import doctested modules by their true name under a namespace package.

    `sybil.python.import_path` derives a module name by walking up from the file
    until it finds a directory without `__init__.py`. `src/galax` is a PEP 420
    namespace directory, so that walk stops one level too deep and yields
    `potential._src.api` instead of `galax.potential._src.api`. Resolving the
    path against `src/` instead gives the true dotted name.

    Sybil does not support namespace packages: simplistix/sybil#59 was closed
    unfixed with "pull request to fix would be welcome". Until that lands, this
    subclass is the supported way to override the behaviour -- `Sybil` takes a
    `document_types` mapping of file extension to `Document` subclass, so no
    patching of Sybil internals is needed.

    Note this is unrelated to pytest's `--import-mode`, which Sybil does not use.
    """

    def import_document(self, example: Example) -> None:
        """Import the document's source file, resolving names against `src/`."""
        path = Path(self.path).resolve()
        try:
            relative = path.relative_to(_SRC)
        except ValueError:  # outside src/ -- let Sybil do its usual thing
            super().import_document(example)
            return

        parts = relative.parts[:-1]
        if relative.name != "__init__.py":
            parts += (relative.stem,)

        module = importlib.import_module(".".join(parts))
        self.namespace.update(module.__dict__)
        self.pop_evaluator(self.import_document)
        raise NotEvaluated


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
    document_types={".py": NamespacePackageDocument},
)
rst_docs = Sybil(  # TODO: deprecate
    parsers=[
        rest.DocTestParser(optionflags=optionflags),
        rest.PythonCodeBlockParser(),
        rest.SkipParser(),
    ],
    patterns=["*.rst", "*.py"],
    document_types={".py": NamespacePackageDocument},
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

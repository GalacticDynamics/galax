"""Doctest configuration."""

from __future__ import annotations

from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

from sybil import Sybil
from sybil.parsers.rest import DocTestParser, PythonCodeBlockParser, SkipParser

pytest_collect_file = Sybil(
    parsers=[
        DocTestParser(optionflags=NORMALIZE_WHITESPACE | ELLIPSIS),
        PythonCodeBlockParser(),
        SkipParser(),
    ],
    patterns=["*.rst", "*.py"],
).pytest()


try:
    import gala  # noqa: F401
except ImportError:
    HAS_GALA = False
else:
    HAS_GALA = True


collect_ignore = []
if not HAS_GALA:
    collect_ignore.append("src/galax/coordinates/_compat.py")
    collect_ignore.append("src/galax/dynamics/_compat.py")
    collect_ignore.append("src/galax/potential/_potential/io/_gala.py")

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

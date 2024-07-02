"""Doctest configuration."""

from __future__ import annotations

from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from importlib.util import find_spec

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


# TODO: via separate optional_deps package
HAS_ASTROPY = find_spec("astropy") is not None
HAS_GALA = find_spec("gala") is not None

collect_ignore = []
if not HAS_ASTROPY:
    collect_ignore.append("src/galax/_galax_interop_astropy.py")
if not HAS_GALA:
    collect_ignore.append("src/galax/_galax_interop_gala.py")
    collect_ignore.append("src/galax/potential/_potential/io/_gala.py")

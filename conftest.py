"""Doctest configuration."""

from __future__ import annotations

from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

from sybil import Sybil
from sybil.parsers.rest import DocTestParser, PythonCodeBlockParser, SkipParser

from optional_dependencies import OptionalDependencyEnum, auto
from optional_dependencies.utils import chain_checks, get_version, is_installed

pytest_collect_file = Sybil(
    parsers=[
        DocTestParser(optionflags=NORMALIZE_WHITESPACE | ELLIPSIS),
        PythonCodeBlockParser(),
        SkipParser(),
    ],
    patterns=["*.rst", "*.py"],
).pytest()


class OptDeps(OptionalDependencyEnum):
    """Optional dependencies for ``galax``."""

    ASTROPY = auto()
    GALA = chain_checks(get_version("gala"), is_installed("gala.dynamics"))
    GALPY = auto()
    MATPLOTLIB = auto()


collect_ignore_glob = []
if not OptDeps.ASTROPY.installed:
    collect_ignore_glob.append("src/galax/_interop/galax_interop_astropy/*")
if not OptDeps.GALA.installed:
    collect_ignore_glob.append("src/galax/_interop/galax_interop_gala/*")
if not OptDeps.GALPY.installed:
    collect_ignore_glob.append("src/galax/_interop/galax_interop_galpy/*")

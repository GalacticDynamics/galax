"""Doctest configuration."""

from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

from sybil import Sybil
from sybil.parsers import myst, rest
from sybil.sybil import SybilCollection

from optional_dependencies import OptionalDependencyEnum, auto
from optional_dependencies.utils import chain_checks, get_version, is_installed

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
    collect_ignore_glob.append("src/galax/_interop/galax_interop_astropy/*")
if not OptDeps.GALA.installed:
    collect_ignore_glob.append("src/galax/_interop/galax_interop_gala/*")
if not OptDeps.GALPY.installed:
    collect_ignore_glob.append("src/galax/_interop/galax_interop_galpy/*")

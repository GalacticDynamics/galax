"""The doctest module-name shim must handle every source root.

Sybil derives a module name by walking up until a directory lacks
`__init__.py`. Under a PEP 420 namespace package that stops too deep, so
`conftest` resolves paths against the source roots instead. With the split
there is more than one root.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))
import conftest
from conftest import _module_name_for

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # The root src/ tree
        ("src/galax/potential/_src/api.py", "galax.potential._src.api"),
        ("src/galax/coordinates/__init__.py", "galax.coordinates"),
        # A packages/ tree: the dotted directory name must not leak into the
        # module name, and the walk must not stop at `interop`.
        (
            "packages/galax.interop.gala/src/galax/interop/gala/potential.py",
            "galax.interop.gala.potential",
        ),
        (
            "packages/galax.interop.astropy/src/galax/interop/astropy/__init__.py",
            "galax.interop.astropy",
        ),
    ],
)
def test_resolves_against_every_source_root(
    path: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `_SRC_ROOTS` globs `packages/*/src` at import, which only finds
    # directories that exist. Inject the roots the glob would find, so these
    # cases hold before (and independently of) any distribution being created.
    roots = tuple(
        ROOT / "packages" / d / "src"
        for d in ("galax.interop.gala", "galax.interop.astropy")
    )
    monkeypatch.setattr(conftest, "_SRC_ROOTS", (*conftest._SRC_ROOTS, *roots))
    assert _module_name_for(ROOT / path) == expected


def test_returns_none_outside_any_source_root() -> None:
    """Sybil's own behaviour must still apply to docs and tests."""
    assert _module_name_for(ROOT / "docs" / "getting_started.rst") is None

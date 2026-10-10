"""The doctest module-name shim must handle every source root.

Sybil derives a module name by walking up until a directory lacks
`__init__.py`. Under a PEP 420 namespace package that stops too deep, so
`conftest` resolves paths against the source roots instead. With the split
there is more than one root.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]

# Load the root conftest by location: putting the repo root on `sys.path` would
# make `docs`, `packages`, `src`, ... importable as top-level names.
_spec = importlib.util.spec_from_file_location(
    "galax_root_conftest", ROOT / "conftest.py"
)
conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conftest)
_module_name_for = conftest._module_name_for


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # A portion's nested module: the walk must not stop at the portion
        # name. There is no root `src/` tree any more -- every source root is
        # a `packages/*/src`.
        (
            "packages/galax.dynamics/src/galax/dynamics/_src/api.py",
            "galax.dynamics._src.api",
        ),
        (
            "packages/galax.coordinates/src/galax/coordinates/__init__.py",
            "galax.coordinates",
        ),
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
    # cases hold independently of which distributions are present.
    roots = tuple(
        ROOT / "packages" / d / "src"
        for d in (
            "galax.coordinates",
            "galax.dynamics",
            "galax.interop.gala",
            "galax.interop.astropy",
        )
    )
    monkeypatch.setattr(conftest, "_SRC_ROOTS", (*conftest._SRC_ROOTS, *roots))
    assert _module_name_for(ROOT / path) == expected


def test_returns_none_outside_any_source_root() -> None:
    """Sybil's own behaviour must still apply to docs and tests."""
    assert _module_name_for(ROOT / "docs" / "getting_started.rst") is None

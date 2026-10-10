"""A module must install the jaxtyping import hook for *itself*.

`install_import_hook(name)` instruments only the modules imported inside its
`with` block whose names fall under `name`. Point it at the wrong name and the
block's imports are simply never instrumented -- no error, no warning, the
module just opts out of runtime typechecking.

Three modules were pointed at the wrong name, two of them at names that do not
exist at all (`galax.dynamics.solve`, `galax.dynamics.dynamics`), and nothing
noticed: the hook is a no-op unless `GALAX_ENABLE_RUNTIME_TYPECHECKING` is set,
and per #918 that has never actually happened in CI.
"""

import ast
import pathlib

from collections.abc import Iterator
from types import ModuleType

import pytest

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp

PORTIONS: tuple[ModuleType, ...] = (gc, gp, gd)


def _hook_targets(path: pathlib.Path, /) -> Iterator[str]:
    """Yield the literal target of every `install_import_hook(...)` call."""
    # Explicit utf-8 and `filename`: `read_text()` would decode with the
    # platform default, which on the Windows leg of the matrix is not utf-8,
    # and a nameless `ast.parse` reports a syntax error as "<unknown>".
    source = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "install_import_hook"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            yield node.args[0].value


def _hooks() -> list[tuple[str, pathlib.Path, str]]:
    """Return `(expected target, file, actual target)` for every hook call."""
    found: list[tuple[str, pathlib.Path, str]] = []
    for portion in PORTIONS:
        pkg = pathlib.Path(portion.__file__ or "").parent
        for path in sorted(pkg.glob("*.py")):
            expected = (
                portion.__name__
                if path.name == "__init__.py"
                else f"{portion.__name__}.{path.stem}"
            )
            found.extend((expected, path, t) for t in _hook_targets(path))
    return found


def test_the_scan_finds_hooks() -> None:
    """Guard the guard: an AST walk that matches nothing would prove nothing."""
    assert len(_hooks()) >= 7


@pytest.mark.parametrize(
    ("expected", "path", "actual"),
    _hooks(),
    ids=lambda v: v.name if isinstance(v, pathlib.Path) else str(v),
)
def test_hook_targets_its_own_module(
    expected: str, path: pathlib.Path, actual: str
) -> None:
    """A hook naming anything but its own module instruments nothing."""
    assert actual == expected, f"{path.name} hooks {actual!r}, should be {expected!r}"
